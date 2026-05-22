"""Tests extractor that orchestrates extraction of all image books."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from tqdm import tqdm

from digitex.extractors.base import BaseExtractor, ExtractionResult
from digitex.extractors.book_extractor import BookExtractor
from digitex.extractors.exceptions import DirectoryNotFoundError
from digitex.extractors.progress import JSONProgressTracker, ProgressTracker

if TYPE_CHECKING:
    from pathlib import Path

    from digitex.extractors.conflict_resolution import ConflictResolver

logger = structlog.get_logger()

PROGRESS_FILE = "progress.json"


class TestsExtractor(BaseExtractor):
    """Orchestrates extraction of question images from all image books."""

    def __init__(
        self,
        model_path: Path,
        image_format: str,
        question_max_width: int,
        question_max_height: int,
        books_dir: Path,
        extraction_dir: Path,
        data_dir: Path | None = None,
        progress_tracker: ProgressTracker | None = None,
        book_extractor: BookExtractor | None = None,
        on_conflict: ConflictResolver | None = None,
    ) -> None:
        self.books_dir = books_dir
        self.extraction_dir = extraction_dir
        self.data_dir = data_dir or extraction_dir.parent / "data"

        self._progress_tracker = progress_tracker or JSONProgressTracker(
            self.data_dir / PROGRESS_FILE
        )

        self._book_extractor = book_extractor or BookExtractor(
            model_path=model_path,
            image_format=image_format,
            question_max_width=question_max_width,
            question_max_height=question_max_height,
            on_conflict=on_conflict,
        )

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

        accumulated = ExtractionResult.success_result(
            metadata={"subject": subject, "years": len(year_dirs)}
        )

        for year_dir in tqdm(year_dirs, desc=f"Extracting {subject}"):
            year = year_dir.name

            if self._progress_tracker.is_completed(subject, year):
                logger.info("Skipping, already extracted", subject=subject, year=year)
                accumulated = accumulated.merge(
                    ExtractionResult.success_result(skipped=1)
                )
                continue

            output_dir = self.extraction_dir / subject / year
            book_result = self._book_extractor.extract(year_dir, output_dir)
            # Carry per-book errors and processed counts up to the caller.
            # We treat processed=1 for the year as long as it didn't fail
            # catastrophically; per-page failures live in book_result.errors.
            accumulated = accumulated.merge(
                ExtractionResult(
                    success=book_result.success,
                    processed=1,
                    errors=book_result.errors,
                    warnings=book_result.warnings,
                )
            )

            if book_result.success:
                self._progress_tracker.mark_completed(subject, year)
                self._progress_tracker.save()

        return accumulated

    def get_progress_tracker(self) -> ProgressTracker:
        """Get the progress tracker instance."""
        return self._progress_tracker

    def clear_progress(self) -> None:
        """Clear all progress tracking."""
        self._progress_tracker.clear()
