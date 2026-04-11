"""Tests for the Extractors module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from digitex.extractors.book_extractor import BookExtractor
from digitex.extractors.page_extractor import (
    OCR_LANGUAGE,
    Detection,
    PageExtractor,
)
from digitex.extractors.tests_extractor import PROGRESS_FILE, TestsExtractor


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
        with pytest.raises(FileNotFoundError, match="Image directory not found"):
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
        assert result is None

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
        mock_page_extract.return_value = (1, "A", 1)

        extractor = BookExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        output_dir = tmp_path / "output"
        extractor.extract(tmp_path, output_dir)

        assert output_dir.exists()
        mock_page_extract.assert_called_once()


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

        extractor._crop_and_save(image, polygon, output_path)

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
        mock_extract_text.return_value = "Часть А"
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
        """Test _detect returns detections sorted by position (top-to-bottom, left-to-right)."""
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
        assert extractor._progress_path == Path("extraction") / PROGRESS_FILE
        assert extractor._book_extractor is not None

    def test_load_completed_empty(self, tmp_path: Path) -> None:
        """Test _load_completed returns empty dict when file doesn't exist."""
        extractor = TestsExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            books_dir=tmp_path,
            extraction_dir=tmp_path / "extraction",
        )
        result = extractor._load_completed()
        assert result == {}

    def test_load_completed_with_data(self, tmp_path: Path) -> None:
        """Test _load_completed parses existing progress file."""
        progress_file = tmp_path / PROGRESS_FILE
        progress_file.write_text('{"math": ["2020", "2021"], "physics": ["2019"]}')

        extractor = TestsExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            books_dir=tmp_path,
            extraction_dir=tmp_path,
        )
        result = extractor._load_completed()
        assert result == {"math": {"2020", "2021"}, "physics": {"2019"}}

    def test_save_completed(self, tmp_path: Path) -> None:
        """Test _save_completed writes progress file."""
        extractor = TestsExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            books_dir=tmp_path,
            extraction_dir=tmp_path,
        )
        completed = {"math": {"2020", "2021"}, "physics": {"2019"}}
        extractor._save_completed(completed)

        assert extractor._progress_path.exists()
        content = extractor._progress_path.read_text()
        assert "math" in content
        assert "2020" in content
        assert "2021" in content
        assert "physics" in content
        assert "2019" in content

    def test_is_completed(self, tmp_path: Path) -> None:
        """Test _is_completed correctly checks completion status."""
        extractor = TestsExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            books_dir=tmp_path,
            extraction_dir=tmp_path,
        )
        completed = {"math": {"2020", "2021"}, "physics": {"2019"}}

        assert extractor._is_completed(completed, "math", "2020") is True
        assert extractor._is_completed(completed, "math", "2022") is False
        assert extractor._is_completed(completed, "biology", "2020") is False

    def test_extract_all_raises_on_missing_books_dir(self, tmp_path: Path) -> None:
        """Test extract_all raises FileNotFoundError for missing books directory."""
        extractor = TestsExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            books_dir=tmp_path / "nonexistent",
            extraction_dir=tmp_path / "extraction",
        )
        with pytest.raises(FileNotFoundError, match="Books directory not found"):
            extractor.extract_all()

    def test_extract_all_no_subjects(self, tmp_path: Path) -> None:
        """Test extract_all with empty books directory."""
        extractor = TestsExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            books_dir=tmp_path,
            extraction_dir=tmp_path / "extraction",
        )
        result = extractor.extract_all()
        assert result is None

    def test_extract_all_skips_completed(self, tmp_path: Path) -> None:
        """Test extract_all skips already completed extractions."""
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
        progress_file = extraction_dir / PROGRESS_FILE
        progress_file.write_text('{"math": ["2020"]}')

        extractor = TestsExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
            books_dir=books_dir,
            extraction_dir=extraction_dir,
        )

        with patch.object(extractor._book_extractor, "extract") as mock_extract:
            extractor.extract_all()
            mock_extract.assert_not_called()
