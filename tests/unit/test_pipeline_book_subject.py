"""Tests for the book / subject extractors and the shared ExtractionResult."""

from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from digitex.domain.placement import PageExtractionState, QuestionPlacement
from digitex.pipeline.base import ExtractionResult
from digitex.pipeline.book import BookExtractor
from digitex.pipeline.exceptions import DirectoryNotFoundError, ReviewAborted
from digitex.pipeline.page import PageExtractor
from digitex.pipeline.pieces import HeldPiece, PageCarry
from digitex.pipeline.subject import PROGRESS_FILE, SubjectExtractor


class _RecordingPageExtractor:
    """Stands in for PageExtractor: records page names, optionally fails.

    Each page commits one question, so the state handed in comes back advanced
    exactly as far as the number of pages processed — that is what lets the
    cross-page numbering test observe the state being threaded.
    """

    def __init__(
        self,
        fail_on: str | None = None,
        abort_on: str | None = None,
        collide_on: str | None = None,
        hold_on: str | None = None,
    ) -> None:
        self.pages: list[str] = []
        self.questions_on_arrival: list[int] = []
        self.positions: list[tuple[int, int]] = []
        self.carried: list[int] = []
        self._fail_on = fail_on
        self._abort_on = abort_on
        self._collide_on = collide_on
        self._hold_on = hold_on

    def extract(
        self,
        image: Image.Image,
        output_dir: Path,
        state: PageExtractionState,
        page_number: int = 0,
        page_count: int = 0,
        carry: PageCarry | None = None,
    ) -> list[QuestionPlacement]:
        # BookExtractor opens pages from disk, so PIL knows the filename —
        # though only ImageFile declares it, hence the defaulted lookup.
        name = Path(getattr(image, "filename", "")).name
        if name == self._abort_on:
            raise ReviewAborted(name)
        if name == self._fail_on:
            raise ValueError("unreadable page")
        self.pages.append(name)
        self.questions_on_arrival.append(state.question)
        self.positions.append((page_number, page_count))
        self.carried.append(len(carry.pieces) if carry else 0)
        if carry is not None and name == self._hold_on:
            carry.hold([HeldPiece(image=image, page_name=name)])
        state.next_question()
        state.commit_question()
        if name == self._collide_on:
            return [QuestionPlacement(option=1, part="A", number=1)]
        return []

    def as_page_extractor(self) -> PageExtractor:
        """This fake satisfies PageExtractor's contract structurally."""
        return cast("PageExtractor", self)


def _write_page(image_dir: Path, name: str) -> None:
    Image.new("RGB", (10, 10), color="white").save(image_dir / name)


class TestBookExtractor:
    def test_extract_raises_on_missing_dir(self, tmp_path: Path) -> None:
        extractor = BookExtractor(_RecordingPageExtractor().as_page_extractor())
        with pytest.raises(DirectoryNotFoundError, match="Directory not found"):
            extractor.extract(tmp_path / "nonexistent", tmp_path / "output")

    def test_extract_no_images_warns(self, tmp_path: Path) -> None:
        extractor = BookExtractor(_RecordingPageExtractor().as_page_extractor())
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
        extractor = BookExtractor(pages.as_page_extractor())
        output_dir = tmp_path / "output"

        result = extractor.extract(image_dir, output_dir)

        assert output_dir.exists()
        assert pages.pages == ["page_1.jpg", "page_2.jpg", "page_10.jpg"]
        assert result.success
        assert result.processed == 3

    def test_each_page_is_told_its_place_in_the_book(self, tmp_path: Path) -> None:
        """Numbered in reading order, so a reviewer can say how far it has got."""
        image_dir = tmp_path / "book"
        image_dir.mkdir()
        for name in ("page_10.jpg", "page_2.jpg", "page_1.jpg"):
            _write_page(image_dir, name)
        pages = _RecordingPageExtractor()
        extractor = BookExtractor(pages.as_page_extractor())

        extractor.extract(image_dir, tmp_path / "output")

        assert pages.positions == [(1, 3), (2, 3), (3, 3)]

    def test_extract_counts_failed_pages_and_continues(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "book"
        image_dir.mkdir()
        _write_page(image_dir, "page_1.jpg")
        _write_page(image_dir, "page_2.jpg")
        pages = _RecordingPageExtractor(fail_on="page_1.jpg")
        extractor = BookExtractor(pages.as_page_extractor())

        result = extractor.extract(image_dir, tmp_path / "output")

        assert result.success  # partial success — caller inspects errors
        assert result.processed == 1
        assert len(result.errors) == 1
        assert "page_1.jpg" in result.errors[0]
        assert result.failed == 1

    def test_a_collision_is_a_warning_not_an_error(self, tmp_path: Path) -> None:
        """A kept-existing file must reach the caller without failing the book.

        Counting it as an error would break resume: replaying an unfinished
        year collides with its own output, and a year with errors is never
        marked completed.
        """
        image_dir = tmp_path / "book"
        image_dir.mkdir()
        _write_page(image_dir, "page_1.jpg")
        pages = _RecordingPageExtractor(collide_on="page_1.jpg")
        extractor = BookExtractor(pages.as_page_extractor())

        result = extractor.extract(image_dir, tmp_path / "output")

        assert result.success
        assert result.errors == []
        assert result.warnings == [
            "page_1.jpg: 1/A/1 already extracted, kept the existing image"
        ]

    def test_question_numbering_continues_across_pages(self, tmp_path: Path) -> None:
        """One state spans the book, so page 2 does not restart at question 1."""
        image_dir = tmp_path / "book"
        image_dir.mkdir()
        for name in ("page_1.jpg", "page_2.jpg", "page_3.jpg"):
            _write_page(image_dir, name)
        pages = _RecordingPageExtractor()
        extractor = BookExtractor(pages.as_page_extractor())

        extractor.extract(image_dir, tmp_path / "output")

        # Each page sees the count the previous ones left behind, not zero.
        assert pages.pages == ["page_1.jpg", "page_2.jpg", "page_3.jpg"]
        assert pages.questions_on_arrival == [0, 1, 2]

    def test_a_failed_page_leaves_the_state_advanced_for_the_next_one(
        self, tmp_path: Path
    ) -> None:
        """A mid-book failure does not reset numbering for the pages after it."""
        image_dir = tmp_path / "book"
        image_dir.mkdir()
        for name in ("page_1.jpg", "page_2.jpg", "page_3.jpg"):
            _write_page(image_dir, name)
        pages = _RecordingPageExtractor(fail_on="page_2.jpg")
        extractor = BookExtractor(pages.as_page_extractor())

        result = extractor.extract(image_dir, tmp_path / "output")

        assert pages.pages == ["page_1.jpg", "page_3.jpg"]
        assert pages.questions_on_arrival == [0, 1]
        assert result.processed == 2
        assert len(result.errors) == 1

    def test_an_aborted_review_stops_the_book_instead_of_counting_a_failure(
        self, tmp_path: Path
    ) -> None:
        """A run the reviewer walked away from must not look like a finished one."""
        image_dir = tmp_path / "book"
        image_dir.mkdir()
        for name in ("page_1.jpg", "page_2.jpg", "page_3.jpg"):
            _write_page(image_dir, name)
        pages = _RecordingPageExtractor(abort_on="page_2.jpg")
        extractor = BookExtractor(pages.as_page_extractor())

        with pytest.raises(ReviewAborted, match=r"page_2\.jpg"):
            extractor.extract(image_dir, tmp_path / "output")

        assert pages.pages == ["page_1.jpg"]


class TestBookExtractorPieces:
    """A question printed across a page break is one book's business."""

    def test_a_piece_reaches_the_page_that_finishes_it(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "book"
        image_dir.mkdir()
        for name in ("1.jpg", "2.jpg"):
            _write_page(image_dir, name)
        pages = _RecordingPageExtractor(hold_on="1.jpg")

        BookExtractor(pages.as_page_extractor()).extract(image_dir, tmp_path / "output")

        # Nothing carried into the first page, the first page's piece into the
        # second — one carry, threaded.
        assert pages.carried == [0, 1]

    def test_a_piece_left_over_at_the_end_of_a_book_is_reported(
        self, tmp_path: Path
    ) -> None:
        """Nothing was written for it, so a silent drop would lose a question."""
        image_dir = tmp_path / "book"
        image_dir.mkdir()
        _write_page(image_dir, "1.jpg")
        pages = _RecordingPageExtractor(hold_on="1.jpg")

        result = BookExtractor(pages.as_page_extractor()).extract(
            image_dir, tmp_path / "output"
        )

        assert result.success
        assert result.warnings == [
            "1.jpg: a question piece was left unfinished, nothing was written for it"
        ]

    def test_no_carry_survives_into_the_next_book(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "book"
        image_dir.mkdir()
        _write_page(image_dir, "1.jpg")
        pages = _RecordingPageExtractor(hold_on="1.jpg")
        extractor = BookExtractor(pages.as_page_extractor())

        extractor.extract(image_dir, tmp_path / "first")
        extractor.extract(image_dir, tmp_path / "second")

        assert pages.carried == [0, 0]


class TestSubjectExtractor:
    def _extractor(self, tmp_path: Path, **overrides) -> SubjectExtractor:
        defaults: dict = {
            "books_dir": tmp_path / "books",
            "extraction_dir": tmp_path / "extraction",
            "book_extractor": BookExtractor(
                _RecordingPageExtractor().as_page_extractor()
            ),
        }
        defaults.update(overrides)
        return SubjectExtractor(**defaults)

    def test_init(self, tmp_path: Path) -> None:
        extractor = self._extractor(tmp_path)
        assert extractor.books_dir == tmp_path / "books"
        assert extractor.extraction_dir == tmp_path / "extraction"

    def test_progress_log_defaults_beside_the_output_tree(self, tmp_path: Path) -> None:
        """``extraction_dir`` is ``.../data/output``; progress lives in ``.../data``."""
        extractor = self._extractor(
            tmp_path, extraction_dir=tmp_path / "extraction" / "data" / "output"
        )
        assert extractor.data_dir == tmp_path / "extraction" / "data"

    def test_extract_fails_on_missing_books_dir(self, tmp_path: Path) -> None:
        result = self._extractor(tmp_path).extract("math")
        assert not result.success
        assert len(result.errors) > 0

    def test_extract_fails_on_unknown_subject(self, tmp_path: Path) -> None:
        (tmp_path / "books").mkdir()
        result = self._extractor(tmp_path).extract("nonexistent")
        assert not result.success
        assert "Subject 'nonexistent' not found" in result.errors[0]

    def test_extract_fails_without_pages_folder(self, tmp_path: Path) -> None:
        (tmp_path / "books" / "math").mkdir(parents=True)
        result = self._extractor(tmp_path).extract("math")
        assert not result.success
        assert "No processed pages folder found" in result.errors[0]

    def test_extract_warns_on_empty_pages_folder(self, tmp_path: Path) -> None:
        (tmp_path / "books" / "math" / "processed" / "pages").mkdir(parents=True)
        result = self._extractor(tmp_path).extract("math")
        assert result.success
        assert result.processed == 0
        assert len(result.warnings) > 0

    def test_extract_skips_completed_years(self, tmp_path: Path) -> None:
        year_dir = tmp_path / "books" / "math" / "processed" / "pages" / "2020"
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
        year_dir = tmp_path / "books" / "math" / "processed" / "pages" / "2020"
        year_dir.mkdir(parents=True)
        _write_page(year_dir, "page_1.jpg")

        data_dir = tmp_path / "data"
        pages = _RecordingPageExtractor()
        extractor = self._extractor(
            tmp_path,
            data_dir=data_dir,
            book_extractor=BookExtractor(pages.as_page_extractor()),
        )

        result = extractor.extract("math")

        assert result.processed == 1
        assert pages.pages == ["page_1.jpg"]
        # Progress persists without the caller asking for a save.
        assert '"2020"' in (data_dir / PROGRESS_FILE).read_text()

    def test_extract_records_a_year_whose_only_issue_was_collisions(
        self, tmp_path: Path
    ) -> None:
        """Collisions are how resuming a year replays already-written pages.

        Blocking ``mark_completed`` on them would leave a resumed year
        unfinished forever; they carry up as warnings instead.
        """
        year_dir = tmp_path / "books" / "math" / "processed" / "pages" / "2020"
        year_dir.mkdir(parents=True)
        _write_page(year_dir, "page_1.jpg")

        data_dir = tmp_path / "data"
        pages = _RecordingPageExtractor(collide_on="page_1.jpg")
        extractor = self._extractor(
            tmp_path,
            data_dir=data_dir,
            book_extractor=BookExtractor(pages.as_page_extractor()),
        )

        result = extractor.extract("math")

        assert result.warnings
        assert '"2020"' in (data_dir / PROGRESS_FILE).read_text()

    def test_extract_does_not_record_a_year_whose_pages_failed(
        self, tmp_path: Path
    ) -> None:
        """A partially-failed book stays retryable.

        BookExtractor reports ``success=True`` alongside per-page errors, and a
        year written to the progress file is skipped on every later run.
        """
        year_dir = tmp_path / "books" / "math" / "processed" / "pages" / "2020"
        year_dir.mkdir(parents=True)
        _write_page(year_dir, "page_1.jpg")
        _write_page(year_dir, "page_2.jpg")

        data_dir = tmp_path / "data"
        pages = _RecordingPageExtractor(fail_on="page_2.jpg")
        extractor = self._extractor(
            tmp_path,
            data_dir=data_dir,
            book_extractor=BookExtractor(pages.as_page_extractor()),
        )

        result = extractor.extract("math")

        assert result.errors
        assert not (data_dir / PROGRESS_FILE).exists()

    def test_extract_does_not_record_a_year_with_no_pages(self, tmp_path: Path) -> None:
        """An empty year directory is "nothing to do", not "done".

        BookExtractor reports success over zero pages, and a year written to
        the progress file is never retried — so scans copied in afterwards
        would be skipped forever.
        """
        (tmp_path / "books" / "math" / "processed" / "pages" / "2020").mkdir(
            parents=True
        )
        data_dir = tmp_path / "data"

        result = self._extractor(tmp_path, data_dir=data_dir).extract("math")

        assert result.success
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
