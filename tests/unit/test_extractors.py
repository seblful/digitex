"""Tests for the book / tests extractors and the shared ExtractionResult."""

from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from PIL import Image

from digitex.extractors.base import ExtractionResult
from digitex.extractors.book_extractor import BookExtractor
from digitex.extractors.exceptions import DirectoryNotFoundError
from digitex.extractors.page_extractor import PageExtractionState, PageExtractor
from digitex.extractors.tests_extractor import PROGRESS_FILE, TestsExtractor


class _RecordingPageExtractor:
    """Stands in for PageExtractor: records page names, optionally fails."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.pages: list[str] = []
        self._fail_on = fail_on

    def extract(
        self,
        image: Image.Image,
        output_dir: Path,
        state: PageExtractionState,
        source_image_name: str = "",
    ) -> PageExtractionState:
        if source_image_name == self._fail_on:
            raise ValueError("unreadable page")
        self.pages.append(source_image_name)
        return state

    def as_page_extractor(self) -> PageExtractor:
        """This fake satisfies PageExtractor's contract structurally."""
        return cast("PageExtractor", self)


def _write_page(image_dir: Path, name: str) -> None:
    Image.new("RGB", (10, 10), color="white").save(image_dir / name)


class TestBookExtractor:
    def test_init_builds_page_extractor_from_config(self) -> None:
        extractor = BookExtractor(
            model_path=Path("model.pt"),
            image_format="jpg",
            question_max_width=2000,
            question_max_height=2000,
        )
        assert extractor._page_extractor.image_format == "jpg"
        assert extractor._page_extractor.question_max_width == 2000
        assert extractor._page_extractor.question_max_height == 2000

    def test_extract_raises_on_missing_dir(self, tmp_path: Path) -> None:
        extractor = BookExtractor(model_path=Path("model.pt"))
        with pytest.raises(DirectoryNotFoundError, match="Directory not found"):
            extractor.extract(tmp_path / "nonexistent", tmp_path / "output")

    def test_extract_no_images_warns(self, tmp_path: Path) -> None:
        extractor = BookExtractor(model_path=Path("model.pt"))
        result = extractor.extract(tmp_path, tmp_path / "output")
        assert result.success
        assert result.processed == 0
        assert result.warnings == ["No images found"]

    def test_extract_processes_pages_in_natural_order(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "book"
        image_dir.mkdir()
        for name in ("page_10.jpg", "page_2.jpg", "page_1.jpg"):
            _write_page(image_dir, name)
        pages = _RecordingPageExtractor()
        extractor = BookExtractor(
            model_path=Path("model.pt"), page_extractor=pages.as_page_extractor()
        )
        output_dir = tmp_path / "output"

        result = extractor.extract(image_dir, output_dir)

        assert output_dir.exists()
        assert pages.pages == ["page_1.jpg", "page_2.jpg", "page_10.jpg"]
        assert result.success
        assert result.processed == 3

    def test_extract_counts_failed_pages_and_continues(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "book"
        image_dir.mkdir()
        _write_page(image_dir, "page_1.jpg")
        _write_page(image_dir, "page_2.jpg")
        pages = _RecordingPageExtractor(fail_on="page_1.jpg")
        extractor = BookExtractor(
            model_path=Path("model.pt"), page_extractor=pages.as_page_extractor()
        )

        result = extractor.extract(image_dir, tmp_path / "output")

        assert result.success  # partial success — caller inspects errors
        assert result.processed == 1
        assert len(result.errors) == 1
        assert "page_1.jpg" in result.errors[0]
        assert result.metadata == {"failed": 1}


class TestTestsExtractor:
    def _extractor(self, tmp_path: Path, **overrides) -> TestsExtractor:
        defaults: dict = {
            "model_path": Path("model.pt"),
            "image_format": "jpg",
            "question_max_width": 2000,
            "question_max_height": 2000,
            "books_dir": tmp_path / "books",
            "extraction_dir": tmp_path / "extraction",
        }
        defaults.update(overrides)
        return TestsExtractor(**defaults)

    def test_init(self, tmp_path: Path) -> None:
        extractor = self._extractor(tmp_path)
        assert extractor.books_dir == tmp_path / "books"
        assert extractor.extraction_dir == tmp_path / "extraction"
        assert extractor.get_progress_tracker() is not None

    def test_extract_fails_on_missing_books_dir(self, tmp_path: Path) -> None:
        result = self._extractor(tmp_path).extract("math")
        assert not result.success
        assert len(result.errors) > 0

    def test_extract_fails_on_unknown_subject(self, tmp_path: Path) -> None:
        (tmp_path / "books").mkdir()
        result = self._extractor(tmp_path).extract("nonexistent")
        assert not result.success
        assert "Subject 'nonexistent' not found" in result.errors[0]

    def test_extract_fails_without_images_folder(self, tmp_path: Path) -> None:
        (tmp_path / "books" / "math").mkdir(parents=True)
        result = self._extractor(tmp_path).extract("math")
        assert not result.success
        assert "No images folder found" in result.errors[0]

    def test_extract_warns_on_empty_images_folder(self, tmp_path: Path) -> None:
        (tmp_path / "books" / "math" / "images").mkdir(parents=True)
        result = self._extractor(tmp_path).extract("math")
        assert result.success
        assert result.processed == 0
        assert len(result.warnings) > 0

    def test_extract_skips_completed_years(self, tmp_path: Path) -> None:
        year_dir = tmp_path / "books" / "math" / "images" / "2020"
        year_dir.mkdir(parents=True)
        (year_dir / "page1.jpg").touch()
        (year_dir / "page2.jpg").touch()

        (tmp_path / "extraction").mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / PROGRESS_FILE).write_text('{"math": ["2020"]}')

        extractor = self._extractor(tmp_path, data_dir=data_dir)

        with patch.object(extractor._book_extractor, "extract") as mock_extract:
            result = extractor.extract("math")

        mock_extract.assert_not_called()
        assert result.skipped == 1
        assert result.processed == 0


class TestExtractionResult:
    def test_success_result(self) -> None:
        result = ExtractionResult.success_result(
            processed=10, skipped=2, warnings=["Warning 1"]
        )
        assert result.success is True
        assert result.processed == 10
        assert result.skipped == 2
        assert result.warnings == ["Warning 1"]
        assert result.errors == []

    def test_failure_result(self) -> None:
        result = ExtractionResult.failure_result(
            errors=["Error 1", "Error 2"], processed=5
        )
        assert result.success is False
        assert result.processed == 5
        assert result.errors == ["Error 1", "Error 2"]
        assert result.warnings == []

    def test_merge_results(self) -> None:
        result1 = ExtractionResult.success_result(processed=10, warnings=["Warning 1"])
        result2 = ExtractionResult.success_result(processed=5, warnings=["Warning 2"])

        merged = result1.merge(result2)

        assert merged.processed == 15
        assert merged.warnings == ["Warning 1", "Warning 2"]
        assert merged.success is True
