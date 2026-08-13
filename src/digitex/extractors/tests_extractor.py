"""Tests extractor that orchestrates extraction of all image books."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from tqdm import tqdm

from digitex.extractors.base import ExtractionResult
from digitex.extractors.book_extractor import BookExtractor
from digitex.extractors.exceptions import DirectoryNotFoundError
from digitex.extractors.progress import JSONProgressTracker

if TYPE_CHECKING:
    from pathlib import Path

    from digitex.extractors.base import ExtractionConfig

logger = structlog.get_logger()

PROGRESS_FILE = "progress.json"


class TestsExtractor:
    """Orchestrates extraction of question images from all image books.

    ``extract(subject)`` is the whole interface. Per-year progress is an
    implementation detail: completed years are skipped and newly finished ones
    recorded, with no tracker handed back to the caller.

    ``book_extractor`` is the one injectable collaborator, mirroring
    :class:`BookExtractor`'s own ``page_extractor``. Configuring anything
    deeper — a conflict resolver, a stand-in predictor — means building the
    chain from the bottom and passing the result in here.
    """

    def __init__(
        self,
        config: ExtractionConfig,
        books_dir: Path,
        extraction_dir: Path,
        data_dir: Path | None = None,
        book_extractor: BookExtractor | None = None,
    ) -> None:
        self.books_dir = books_dir
        self.extraction_dir = extraction_dir
        # ``extraction_dir`` is the output tree (``var/extraction/output``), so
        # the progress log belongs beside it in ``var/extraction``.
        self.data_dir = data_dir or extraction_dir.parent

        self._progress = JSONProgressTracker(self.data_dir / PROGRESS_FILE)
        self._book_extractor = book_extractor or BookExtractor(config)

    def _validate_books_dir(self) -> None:
        if not self.books_dir.exists():
            raise DirectoryNotFoundError(self.books_dir)

    def extract(self, subject: str) -> ExtractionResult:
        """Extract question images for a specific subject.

        Per-book failures are merged into the returned result so the caller
        sees an honest count of processed/failed years.
        """
        try:
            self._validate_books_dir()
        except DirectoryNotFoundError as e:
            return ExtractionResult.failure_result(errors=[str(e)])

        subject_dir = self.books_dir / subject

        if not subject_dir.exists():
            return ExtractionResult.failure_result(
                errors=[f"Subject '{subject}' not found in {self.books_dir}"]
            )

        images_dir = subject_dir / "images"

        if not images_dir.exists():
            logger.warning("No images folder found", subject_dir=str(subject_dir))
            return ExtractionResult.failure_result(
                errors=[f"No images folder found for subject '{subject}'"]
            )

        year_dirs = [d for d in images_dir.iterdir() if d.is_dir()]

        if not year_dirs:
            logger.warning("No year folders found", images_dir=str(images_dir))
            return ExtractionResult.success_result(
                processed=0, warnings=[f"No year folders found for subject '{subject}'"]
            )

        accumulated = ExtractionResult.success_result()

        for year_dir in tqdm(year_dirs, desc=f"Extracting {subject}"):
            year = year_dir.name

            if self._progress.is_completed(subject, year):
                logger.info("Skipping, already extracted", subject=subject, year=year)
                accumulated = accumulated.merge(
                    ExtractionResult.success_result(skipped=1)
                )
                continue

            output_dir = self.extraction_dir / subject / year
            book_result = self._book_extractor.extract(year_dir, output_dir)
            # ``processed`` counts years here, not pages, so the book's own
            # count is replaced by 1 — but its failed pages and messages carry
            # up, or the caller would be told a partial year was clean.
            accumulated = accumulated.merge(
                ExtractionResult(
                    success=book_result.success,
                    processed=1,
                    failed=book_result.failed,
                    errors=book_result.errors,
                    warnings=book_result.warnings,
                )
            )

            # Only a clean run over at least one page counts as done —
            # BookExtractor reports partial success, an empty book directory
            # reports success over nothing, and a year marked completed is
            # never retried.
            if book_result.success and not book_result.errors and book_result.processed:
                self._progress.mark_completed(subject, year)

        return accumulated
