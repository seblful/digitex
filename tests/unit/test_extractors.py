"""Tests for the book / tests extractors and the shared ExtractionResult."""

from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from digitex.extractors.base import ExtractionConfig, ExtractionResult
from digitex.extractors.book_extractor import BookExtractor
from digitex.extractors.exceptions import DirectoryNotFoundError
from digitex.extractors.page_extractor import PageExtractionState, PageExtractor
from digitex.extractors.tests_extractor import PROGRESS_FILE, TestsExtractor


def _config() -> ExtractionConfig:
    return ExtractionConfig(model_path=Path("model.pt"))


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
    def test_extract_raises_on_missing_dir(self, tmp_path: Path) -> None:
        extractor = BookExtractor(_config())
        with pytest.raises(DirectoryNotFoundError, match="Directory not found"):
            extractor.extract(tmp_path / "nonexistent", tmp_path / "output")

    def test_extract_no_images_warns(self, tmp_path: Path) -> None:
        extractor = BookExtractor(_config())
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
        extractor = BookExtractor(_config(), page_extractor=pages.as_page_extractor())
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
        extractor = BookExtractor(_config(), page_extractor=pages.as_page_extractor())

        result = extractor.extract(image_dir, tmp_path / "output")

        assert result.success  # partial success — caller inspects errors
        assert result.processed == 1
        assert len(result.errors) == 1
        assert "page_1.jpg" in result.errors[0]
        assert result.metadata == {"failed": 1}


class TestTestsExtractor:
    def _extractor(self, tmp_path: Path, **overrides) -> TestsExtractor:
        defaults: dict = {
            "config": _config(),
            "books_dir": tmp_path / "books",
            "extraction_dir": tmp_path / "extraction",
        }
        defaults.update(overrides)
        return TestsExtractor(**defaults)

    def test_init(self, tmp_path: Path) -> None:
        extractor = self._extractor(tmp_path)
        assert extractor.books_dir == tmp_path / "books"
        assert extractor.extraction_dir == tmp_path / "extraction"

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

        result = self._extractor(tmp_path, data_dir=data_dir).extract("math")

        assert result.skipped == 1
        assert result.processed == 0
        # Nothing was written for the year — the book was never opened.
        assert not (tmp_path / "extraction" / "math").exists()

    def test_extract_records_finished_years_in_the_progress_file(
        self, tmp_path: Path
    ) -> None:
        year_dir = tmp_path / "books" / "math" / "images" / "2020"
        year_dir.mkdir(parents=True)
        _write_page(year_dir, "page_1.jpg")

        data_dir = tmp_path / "data"
        pages = _RecordingPageExtractor()
        extractor = self._extractor(tmp_path, data_dir=data_dir)
        extractor._book_extractor = BookExtractor(
            _config(), page_extractor=pages.as_page_extractor()
        )

        result = extractor.extract("math")

        assert result.processed == 1
        assert pages.pages == ["page_1.jpg"]
        # Progress persists without the caller asking for a save.
        assert '"2020"' in (data_dir / PROGRESS_FILE).read_text()

    def test_extract_does_not_record_a_year_whose_pages_failed(
        self, tmp_path: Path
    ) -> None:
        """A partially-failed book stays retryable.

        BookExtractor reports ``success=True`` alongside per-page errors, and a
        year written to the progress file is skipped on every later run.
        """
        year_dir = tmp_path / "books" / "math" / "images" / "2020"
        year_dir.mkdir(parents=True)
        _write_page(year_dir, "page_1.jpg")
        _write_page(year_dir, "page_2.jpg")

        data_dir = tmp_path / "data"
        pages = _RecordingPageExtractor(fail_on="page_2.jpg")
        extractor = self._extractor(tmp_path, data_dir=data_dir)
        extractor._book_extractor = BookExtractor(
            _config(), page_extractor=pages.as_page_extractor()
        )

        result = extractor.extract("math")

        assert result.errors
        assert not (data_dir / PROGRESS_FILE).exists()


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
