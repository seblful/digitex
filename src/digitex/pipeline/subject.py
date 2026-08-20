"""Tests extractor that orchestrates extraction of all image books."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from tqdm import tqdm

from digitex.domain.corpus import PROCESSED, book_pages_dir
from digitex.pipeline.exceptions import DirectoryNotFoundError
from digitex.pipeline.outcome import (
    SubjectOutcome,
    SubjectRefused,
    SubjectReport,
    YearReport,
)
from digitex.pipeline.progress import JSONProgressTracker

if TYPE_CHECKING:
    from pathlib import Path

    from digitex.pipeline.book import BookExtractor

logger = structlog.get_logger()

PROGRESS_FILE = "progress.json"


class SubjectExtractor:
    """Orchestrates extraction of question images from all image books.

    ``extract(subject)`` is the whole interface. Per-year progress is an
    implementation detail: completed years are skipped and newly finished ones
    recorded, with no tracker handed back to the caller.

    ``book_extractor`` does every year, mirroring :class:`BookExtractor`'s
    own ``page_extractor``. Configuring anything deeper — a reviewer, a
    stand-in predictor — means building the chain from the bottom and passing
    the result in here, so no extraction config reaches this class either.
    """

    def __init__(
        self,
        books_dir: Path,
        extraction_dir: Path,
        book_extractor: BookExtractor,
        data_dir: Path | None = None,
    ) -> None:
        self.books_dir = books_dir
        self.extraction_dir = extraction_dir
        # ``extraction_dir`` is the output tree (``var/extraction/output``), so
        # the progress log belongs beside it in ``var/extraction``.
        self.data_dir = data_dir or extraction_dir.parent

        self._progress = JSONProgressTracker(self.data_dir / PROGRESS_FILE)
        self._book_extractor = book_extractor

    def _validate_books_dir(self) -> None:
        if not self.books_dir.exists():
            raise DirectoryNotFoundError(self.books_dir)

    def extract(self, subject: str) -> SubjectOutcome:
        """Extract question images for a specific subject.

        Every year that ran comes back in the report, whether or not its pages
        all succeeded. A :class:`SubjectRefused` means the run never began —
        distinct from a report holding no years, which means there was nothing
        left to do.
        """
        try:
            self._validate_books_dir()
        except DirectoryNotFoundError as e:
            return SubjectRefused(reason=str(e))

        subject_dir = self.books_dir / subject

        if not subject_dir.exists():
            return SubjectRefused(
                reason=f"Subject '{subject}' not found in {self.books_dir}"
            )

        # The processed variant, never the raw one: the segmentation model is
        # trained on corrected pages, so it has to be shown corrected pages.
        pages_dir = book_pages_dir(self.books_dir, subject, PROCESSED)

        if not pages_dir.exists():
            logger.warning("No pages folder found", subject_dir=str(subject_dir))
            return SubjectRefused(
                reason=f"No processed pages folder found for subject '{subject}';"
                " run preprocess-scans first"
            )

        year_dirs = [d for d in pages_dir.iterdir() if d.is_dir()]

        if not year_dirs:
            logger.warning("No year folders found", pages_dir=str(pages_dir))
            return SubjectReport()

        years: list[YearReport] = []
        skipped: list[str] = []

        for year_dir in tqdm(year_dirs, desc=f"Extracting {subject}"):
            year = year_dir.name

            if self._progress.is_completed(subject, year):
                logger.info("Skipping, already extracted", subject=subject, year=year)
                skipped.append(year)
                continue

            output_dir = self.extraction_dir / subject / year
            book = self._book_extractor.extract(year_dir, output_dir)
            years.append(YearReport(year=year, book=book))

            # A year marked completed is never retried, so the bar is a clean
            # run over at least one page — which is what ``complete`` says.
            if book.complete:
                self._progress.mark_completed(subject, year)

        return SubjectReport(years=tuple(years), skipped=tuple(skipped))
