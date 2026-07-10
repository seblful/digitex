"""Tests for the Extractors module."""

from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from digitex.extractors.base import ExtractionResult
from digitex.extractors.book_extractor import BookExtractor
from digitex.extractors.exceptions import DirectoryNotFoundError
from digitex.extractors.page_extractor import (
    OCR_LANGUAGE,
    PageExtractionState,
    PageExtractor,
    QuestionPlacement,
)
from digitex.extractors.tests_extractor import PROGRESS_FILE, TestsExtractor
from digitex.ml.predictors import SegmentationPredictionResult


class TestBookExtractor:
    """Test suite for BookExtractor class."""

    def test_init(self) -> None:
        """Test BookExtractor initialization."""
        extractor = BookExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        assert extractor._page_extractor is not None
        assert extractor._page_extractor.image_format == "jpg"
        assert extractor._page_extractor.question_max_width == 2000
        assert extractor._page_extractor.question_max_height == 2000

    def test_extract_raises_on_missing_dir(self, tmp_path: Path) -> None:
        """Test that extract raises FileNotFoundError for missing image directory."""
        extractor = BookExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        nonexistent_dir = tmp_path / "nonexistent"
        with pytest.raises(DirectoryNotFoundError, match="Directory not found"):
            extractor.extract(nonexistent_dir, tmp_path / "output")

    def test_extract_no_images(self, tmp_path: Path) -> None:
        """Test extract with directory containing no images."""
        extractor = BookExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        output_dir = tmp_path / "output"
        result = extractor.extract(tmp_path, output_dir)
        assert isinstance(result, ExtractionResult)
        assert result.processed == 0

    @patch("PIL.Image.open")
    @patch("digitex.extractors.page_extractor.PageExtractor.extract")
    def test_extract_creates_output_dir(
        self, mock_page_extract: MagicMock, mock_image_open: MagicMock, tmp_path: Path
    ) -> None:
        """Test that extract creates output directory."""
        image_path = tmp_path / "test_image.jpg"
        image_path.touch()

        img = Image.new("RGB", (100, 100), color="white")
        mock_image_open.return_value = img
        mock_page_extract.return_value = PageExtractionState(
            option=1, part="A", question=1
        )

        extractor = BookExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        output_dir = tmp_path / "output"
        result = extractor.extract(tmp_path, output_dir)

        assert output_dir.exists()
        mock_page_extract.assert_called_once()
        assert isinstance(result, ExtractionResult)


class TestPageExtractionState:
    """Test the question-numbering state machine through its interface."""

    def test_option_marker_advances_sequentially(self) -> None:
        state = PageExtractionState()
        assert state.on_option(1) is True
        assert (state.option, state.part, state.question) == (1, "A", 0)

    def test_non_sequential_option_marker_ignored(self) -> None:
        state = PageExtractionState(option=1, part="B", question=3)
        assert state.on_option(5) is False
        assert state.on_option(None) is False
        assert (state.option, state.part, state.question) == (1, "B", 3)

    def test_part_marker_switches_and_resets_numbering(self) -> None:
        state = PageExtractionState(option=1, part="A", question=7)
        assert state.on_part("B") is True
        assert (state.part, state.question) == ("B", 0)

    def test_same_or_missing_part_marker_ignored(self) -> None:
        state = PageExtractionState(option=1, part="A", question=7)
        assert state.on_part("A") is False
        assert state.on_part(None) is False
        assert state.question == 7

    def test_placements_number_sequentially_after_commit(self) -> None:
        state = PageExtractionState(option=1, part="A")
        assert state.next_question() == QuestionPlacement(option=1, part="A", number=1)
        state.commit_question()
        assert state.next_question() == QuestionPlacement(option=1, part="A", number=2)

    def test_next_question_without_commit_does_not_consume(self) -> None:
        state = PageExtractionState(option=1, part="A")
        assert state.next_question().number == 1
        assert state.next_question().number == 1

    def test_correct_option_moves_and_keeps_numbering(self) -> None:
        state = PageExtractionState(option=1, part="B", question=3)
        assert state.correct_option(2) is True
        assert (state.option, state.part, state.question) == (2, "A", 3)
        assert state.next_question() == QuestionPlacement(option=2, part="A", number=4)

    def test_correct_option_same_option_is_noop(self) -> None:
        state = PageExtractionState(option=1, part="A", question=3)
        assert state.correct_option(1) is False
        assert (state.option, state.part, state.question) == (1, "A", 3)

    def test_full_page_event_sequence(self) -> None:
        state = PageExtractionState()
        state.on_option(1)
        state.on_part("A")
        placements = [state.next_question()]
        state.commit_question()
        placements.append(state.next_question())
        state.commit_question()
        state.on_part("B")
        placements.append(state.next_question())
        state.commit_question()
        assert placements == [
            QuestionPlacement(option=1, part="A", number=1),
            QuestionPlacement(option=1, part="A", number=2),
            QuestionPlacement(option=1, part="B", number=1),
        ]


class _FakePredictor:
    def __init__(self, result: SegmentationPredictionResult) -> None:
        self._result = result

    def predict(self, image: Image.Image) -> SegmentationPredictionResult:
        return self._result


class _FakeTextExtractor:
    def __init__(self, digits: list[int], text: str) -> None:
        self._digits = digits
        self._text = text

    def extract_digits(self, image: Image.Image) -> list[int]:
        return self._digits

    def extract_text(self, image: Image.Image) -> str:
        return self._text


class TestPageExtractorExtract:
    """Behavior tests of extract() through its interface — no YOLO, no OCR."""

    ID2LABEL: ClassVar[dict[int, str]] = {0: "question", 1: "option", 2: "part"}

    def _extractor(
        self,
        result: SegmentationPredictionResult,
        on_conflict=None,
    ) -> PageExtractor:
        return PageExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=50,
            question_max_height=50,
            predictor=_FakePredictor(result),
            text_extractor=_FakeTextExtractor(digits=[1], text="Часть A"),
            on_conflict=on_conflict,
        )

    def test_questions_saved_under_detected_option_and_part(
        self, tmp_path: Path
    ) -> None:
        result = SegmentationPredictionResult(
            ids=[1, 2, 0, 0],
            polygons=[
                [(10, 0), (40, 0), (40, 10), (10, 10)],
                [(10, 20), (40, 20), (40, 30), (10, 30)],
                [(10, 40), (200, 40), (200, 80), (10, 80)],
                [(10, 90), (200, 90), (200, 130), (10, 130)],
            ],
            id2label=self.ID2LABEL,
        )
        image = Image.new("RGB", (300, 300), color="white")

        state = self._extractor(result).extract(image, tmp_path)

        assert (tmp_path / "1" / "A" / "1.jpg").exists()
        assert (tmp_path / "1" / "A" / "2.jpg").exists()
        assert (state.option, state.part, state.question) == (1, "A", 2)

    def test_conflict_with_default_resolver_keeps_existing_file(
        self, tmp_path: Path
    ) -> None:
        result = SegmentationPredictionResult(
            ids=[0],
            polygons=[[(10, 40), (200, 40), (200, 80), (10, 80)]],
            id2label=self.ID2LABEL,
        )
        existing = tmp_path / "1" / "A" / "1.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"original")
        image = Image.new("RGB", (300, 300), color="white")

        state = self._extractor(result).extract(
            image, tmp_path, PageExtractionState(option=1, part="A")
        )

        assert existing.read_bytes() == b"original"
        assert (state.option, state.question) == (1, 1)

    def test_conflict_resolver_correction_moves_question_and_state(
        self, tmp_path: Path
    ) -> None:
        result = SegmentationPredictionResult(
            ids=[0],
            polygons=[[(10, 40), (200, 40), (200, 80), (10, 80)]],
            id2label=self.ID2LABEL,
        )
        existing = tmp_path / "1" / "A" / "1.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"original")
        image = Image.new("RGB", (300, 300), color="white")

        state = self._extractor(result, on_conflict=lambda conflict: 2).extract(
            image, tmp_path, PageExtractionState(option=1, part="A")
        )

        assert not existing.exists()
        assert (tmp_path / "2" / "A" / "1.jpg").exists()
        assert (state.option, state.part, state.question) == (2, "A", 1)


class TestPageExtractor:
    """Test suite for PageExtractor class."""

    def test_init(self) -> None:
        """Test PageExtractor initialization."""
        extractor = PageExtractor(
            model_path=Path("model.pt"),
            image_format="png",
            question_max_width=1000,
            question_max_height=1000,
        )
        assert extractor.model_path == Path("model.pt")
        assert extractor.image_format == "png"
        assert extractor.question_max_width == 1000
        assert extractor.question_max_height == 1000
        assert extractor._predictor is None
        assert extractor._segment_processor is not None
        assert extractor._image_cropper is not None
        assert extractor._text_extractor is not None
        assert extractor._text_extractor.language == OCR_LANGUAGE

    def test_init_with_dependencies(self) -> None:
        """Test PageExtractor initialization with dependency injection."""
        mock_predictor = MagicMock()
        mock_processor = MagicMock()
        mock_cropper = MagicMock()
        mock_text_extractor = MagicMock()

        extractor = PageExtractor(
            model_path=Path("model.pt"),
            predictor=mock_predictor,
            segment_processor=mock_processor,
            image_cropper=mock_cropper,
            text_extractor=mock_text_extractor,
        )

        assert extractor._predictor is mock_predictor
        assert extractor._segment_processor is mock_processor
        assert extractor._image_cropper is mock_cropper
        assert extractor._text_extractor is mock_text_extractor

    def test_get_label_name(self) -> None:
        """Test _get_label_name returns correct label."""
        extractor = PageExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        mock_result = MagicMock()
        mock_result.id2label = {0: "question", 1: "option", 2: "part"}
        assert extractor._get_label_name(mock_result, 0) == "question"
        assert extractor._get_label_name(mock_result, 1) == "option"
        assert extractor._get_label_name(mock_result, 2) == "part"
        assert extractor._get_label_name(mock_result, 99) == "unknown"

    def test_get_polygon_bounding_box(self) -> None:
        """Test _get_polygon_bounding_box returns correct position."""
        extractor = PageExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        polygon = [(10, 20), (50, 15), (45, 80), (5, 75)]
        min_y, min_x = extractor._get_polygon_bounding_box(polygon)
        assert min_y == 15
        assert min_x == 5

    @patch("digitex.extractors.page_extractor.ImageCropper.cut_out_image_by_polygon")
    @patch("digitex.extractors.page_extractor.resize_image")
    @patch("digitex.extractors.page_extractor.SegmentProcessor.process")
    def test_crop_and_save(
        self,
        mock_process: MagicMock,
        mock_resize: MagicMock,
        mock_cut: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test _crop_and_save processes and saves image correctly."""
        extractor = PageExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )

        mock_cropped = Image.new("RGB", (100, 100), color="white")
        mock_resized = Image.new("RGB", (200, 200), color="white")
        mock_processed = Image.new("RGB", (200, 200), color="white")

        mock_cut.return_value = mock_cropped
        mock_resize.return_value = mock_resized
        mock_process.return_value = mock_processed

        image = Image.new("RGB", (300, 300), color="white")
        polygon = [(10, 10), (100, 10), (100, 100), (10, 100)]
        output_path = tmp_path / "output" / "question.png"

        extractor._crop_and_save(image, polygon, output_path, 1, "test.jpg", tmp_path)

        assert output_path.with_suffix(".jpg").exists()
        mock_cut.assert_called_once_with(image, polygon)
        mock_resize.assert_called_once_with(mock_cropped, 2000, 2000)
        mock_process.assert_called_once_with(mock_resized)

    @patch("digitex.extractors.page_extractor.TextExtractor.extract_digits")
    def test_extract_option_number(
        self, mock_extract_digits: MagicMock, tmp_path: Path
    ) -> None:
        """Test _extract_option_number extracts correct option."""
        extractor = PageExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        mock_extract_digits.return_value = [3]
        image = Image.new("RGB", (100, 100), color="white")
        polygon = [(10, 10), (50, 10), (50, 50), (10, 50)]

        result = extractor._extract_option_number(image, polygon)
        assert result == 3

    @patch("digitex.extractors.page_extractor.TextExtractor.extract_digits")
    def test_extract_option_number_no_digits(
        self, mock_extract_digits: MagicMock, tmp_path: Path
    ) -> None:
        """Test _extract_option_number returns None when no digits found."""
        extractor = PageExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        mock_extract_digits.return_value = []
        image = Image.new("RGB", (100, 100), color="white")
        polygon = [(10, 10), (50, 10), (50, 50), (10, 50)]

        result = extractor._extract_option_number(image, polygon)
        assert result is None

    @patch("digitex.extractors.page_extractor.TextExtractor.extract_text")
    def test_extract_part_letter_a(
        self, mock_extract_text: MagicMock, tmp_path: Path
    ) -> None:
        """Test _extract_part_letter extracts 'A' correctly."""
        extractor = PageExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        mock_extract_text.return_value = "Часть A"
        image = Image.new("RGB", (100, 100), color="white")
        polygon = [(10, 10), (50, 10), (50, 50), (10, 50)]

        result = extractor._extract_part_letter(image, polygon)
        assert result == "A"

    @patch("digitex.extractors.page_extractor.TextExtractor.extract_text")
    def test_extract_part_letter_b(
        self, mock_extract_text: MagicMock, tmp_path: Path
    ) -> None:
        """Test _extract_part_letter extracts 'B' correctly."""
        extractor = PageExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        mock_extract_text.return_value = "часть b"
        image = Image.new("RGB", (100, 100), color="white")
        polygon = [(10, 10), (50, 10), (50, 50), (10, 50)]

        result = extractor._extract_part_letter(image, polygon)
        assert result == "B"

    @patch("digitex.extractors.page_extractor.TextExtractor.extract_text")
    def test_extract_part_letter_cyrillic(
        self, mock_extract_text: MagicMock, tmp_path: Path
    ) -> None:
        """Test _extract_part_letter converts Cyrillic to Latin."""
        extractor = PageExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        mock_extract_text.return_value = "Часть Б"
        image = Image.new("RGB", (100, 100), color="white")
        polygon = [(10, 10), (50, 10), (50, 50), (10, 50)]

        result = extractor._extract_part_letter(image, polygon)
        assert result == "B"

    @patch("digitex.extractors.page_extractor.TextExtractor.extract_text")
    def test_extract_part_letter_no_match(
        self, mock_extract_text: MagicMock, tmp_path: Path
    ) -> None:
        """Test _extract_part_letter returns None when no match found."""
        extractor = PageExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        mock_extract_text.return_value = "some text"
        image = Image.new("RGB", (100, 100), color="white")
        polygon = [(10, 10), (50, 10), (50, 50), (10, 50)]

        result = extractor._extract_part_letter(image, polygon)
        assert result is None

    def test_detect_raises_on_no_detections(self) -> None:
        """Test _detect raises ValueError when no detections found."""
        extractor = PageExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = MagicMock(ids=[])
        extractor._predictor = mock_predictor

        image = Image.new("RGB", (100, 100), color="white")

        with pytest.raises(ValueError, match="No detections found on page"):
            extractor._detect(image)

    def test_detect_returns_sorted_detections(self) -> None:
        """Test _detect returns detections sorted by position."""
        extractor = PageExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )

        mock_result = MagicMock()
        mock_result.id2label = {0: "question", 1: "option", 2: "part"}
        mock_result.ids = [2, 0, 1]
        mock_result.polygons = [
            [(100, 100), (150, 100), (150, 150), (100, 150)],
            [(10, 50), (80, 50), (80, 100), (10, 100)],
            [(50, 200), (100, 200), (100, 250), (50, 250)],
        ]

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = mock_result
        extractor._predictor = mock_predictor

        image = Image.new("RGB", (300, 300), color="white")
        detections = extractor._detect(image)

        assert len(detections) == 3
        labels = [d[0] for d in detections]
        assert labels == ["question", "part", "option"]


class TestTestsExtractor:
    """Test suite for TestsExtractor class."""

    def test_init(self) -> None:
        """Test TestsExtractor initialization."""
        extractor = TestsExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            books_dir=Path("books"),
            extraction_dir=Path("extraction"),
        )
        assert extractor.books_dir == Path("books")
        assert extractor.extraction_dir == Path("extraction")
        assert extractor.get_progress_tracker() is not None
        assert extractor._book_extractor is not None

    def test_extract_raises_on_missing_books_dir(self, tmp_path: Path) -> None:
        """Test extract raises FileNotFoundError for missing books directory."""
        extractor = TestsExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            books_dir=tmp_path / "nonexistent",
            extraction_dir=tmp_path / "extraction",
        )
        result = extractor.extract("math")
        assert not result.success
        assert len(result.errors) > 0

    def test_extract_subject_not_found(self, tmp_path: Path) -> None:
        """Test extract returns error when subject not found."""
        books_dir = tmp_path / "books"
        books_dir.mkdir()

        extractor = TestsExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            books_dir=books_dir,
            extraction_dir=tmp_path / "extraction",
        )
        result = extractor.extract("nonexistent")
        assert not result.success
        assert "Subject 'nonexistent' not found" in result.errors[0]

    def test_extract_no_images_folder(self, tmp_path: Path) -> None:
        """Test extract returns error when images folder missing."""
        books_dir = tmp_path / "books"
        books_dir.mkdir()
        subject_dir = books_dir / "math"
        subject_dir.mkdir()

        extractor = TestsExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            books_dir=books_dir,
            extraction_dir=tmp_path / "extraction",
        )
        result = extractor.extract("math")
        assert not result.success
        assert "No images folder found" in result.errors[0]

    def test_extract_no_year_folders(self, tmp_path: Path) -> None:
        """Test extract returns warning when no year folders."""
        books_dir = tmp_path / "books"
        books_dir.mkdir()
        subject_dir = books_dir / "math"
        images_dir = subject_dir / "images"
        images_dir.mkdir(parents=True)

        extractor = TestsExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            books_dir=books_dir,
            extraction_dir=tmp_path / "extraction",
        )
        result = extractor.extract("math")
        assert result.success
        assert result.processed == 0
        assert len(result.warnings) > 0

    def test_extract_skips_completed(self, tmp_path: Path) -> None:
        """Test extract skips already completed years."""
        books_dir = tmp_path / "books"
        books_dir.mkdir()
        subject_dir = books_dir / "math"
        images_dir = subject_dir / "images"
        year_dir = images_dir / "2020"
        year_dir.mkdir(parents=True)

        (year_dir / "page1.jpg").touch()
        (year_dir / "page2.jpg").touch()

        extraction_dir = tmp_path / "extraction"
        extraction_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        progress_file = data_dir / PROGRESS_FILE
        progress_file.write_text('{"math": ["2020"]}')

        extractor = TestsExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            books_dir=books_dir,
            extraction_dir=extraction_dir,
            data_dir=data_dir,
        )

        with patch.object(extractor._book_extractor, "extract") as mock_extract:
            result = extractor.extract("math")
            mock_extract.assert_not_called()
            assert result.skipped == 1
            assert result.processed == 0


class TestExtractionResult:
    """Test suite for ExtractionResult dataclass."""

    def test_success_result(self) -> None:
        """Test creating a success result."""
        result = ExtractionResult.success_result(
            processed=10, skipped=2, warnings=["Warning 1"]
        )
        assert result.success is True
        assert result.processed == 10
        assert result.skipped == 2
        assert result.warnings == ["Warning 1"]
        assert result.errors == []

    def test_failure_result(self) -> None:
        """Test creating a failure result."""
        result = ExtractionResult.failure_result(
            errors=["Error 1", "Error 2"], processed=5
        )
        assert result.success is False
        assert result.processed == 5
        assert result.errors == ["Error 1", "Error 2"]
        assert result.warnings == []

    def test_merge_results(self) -> None:
        """Test merging two results."""
        result1 = ExtractionResult.success_result(processed=10, warnings=["Warning 1"])
        result2 = ExtractionResult.success_result(processed=5, warnings=["Warning 2"])

        merged = result1.merge(result2)

        assert merged.processed == 15
        assert merged.warnings == ["Warning 1", "Warning 2"]
        assert merged.success is True
