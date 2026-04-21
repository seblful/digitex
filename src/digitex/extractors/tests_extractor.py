"""Tests extractor that orchestrates extraction of all image books."""

import json
from pathlib import Path

import structlog
from tqdm import tqdm

from digitex.extractors.base import BaseExtractor, ExtractionResult
from digitex.extractors.book_extractor import BookExtractor
from digitex.extractors.exceptions import DirectoryNotFoundError
from digitex.extractors.progress import JSONProgressTracker, ProgressTracker

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
    ) -> None:
        """Initialize the tests extractor.

        Args:
            model_path: Path to YOLO segmentation model.
            image_format: Output image format.
            question_max_width: Maximum width for question images.
            question_max_height: Maximum height for question images.
            books_dir: Directory containing subject folders.
            extraction_dir: Output directory for extracted images.
            data_dir: Directory for progress tracking (default: extraction_dir parent / "data").
            progress_tracker: Optional custom progress tracker.
        """
        self.books_dir = books_dir
        self.extraction_dir = extraction_dir
        self.data_dir = data_dir or extraction_dir.parent / "data"

        self._progress_tracker = progress_tracker or JSONProgressTracker(
            self.data_dir / PROGRESS_FILE
        )

        self._book_extractor = BookExtractor(
            model_path=model_path,
            image_format=image_format,
            question_max_width=question_max_width,
            question_max_height=question_max_height,
        )

    def _validate_prerequisites(self) -> None:
        """Validate that books directory exists."""
        if not self.books_dir.exists():
            raise DirectoryNotFoundError(self.books_dir)

    def extract_all(self) -> ExtractionResult:
        """Extract question images from all subjects in the books directory.

        Returns:
            ExtractionResult with statistics.
        """
        try:
            self._validate_prerequisites()
        except DirectoryNotFoundError as e:
            return ExtractionResult.failure_result(errors=[str(e)])

        subject_dirs = [d for d in self.books_dir.iterdir() if d.is_dir()]

        if not subject_dirs:
            logger.warning("No subject folders found", books_dir=str(self.books_dir))
            return ExtractionResult.success_result(
                processed=0, warnings=["No subject folders found"]
            )

        total_processed = 0
        total_skipped = 0
        warnings: list[str] = []

        for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
            subject = subject_dir.name
            images_dir = subject_dir / "images"

            if not images_dir.exists():
                logger.warning(
                    "No images folder found", subject_dir=str(subject_dir)
                )
                warnings.append(f"No images folder for {subject}")
                continue

            year_dirs = [d for d in images_dir.iterdir() if d.is_dir()]

            if not year_dirs:
                logger.warning("No year folders found", images_dir=str(images_dir))
                warnings.append(f"No year folders for {subject}")
                continue

            for year_dir in tqdm(
                year_dirs, desc=f"Extracting {subject}", leave=False
            ):
                year = year_dir.name

                if self._progress_tracker.is_completed(subject, year):
                    logger.info("Skipping, already extracted", subject=subject, year=year)
                    total_skipped += 1
                    continue

                output_dir = self.extraction_dir / subject / year
                self._book_extractor.extract(year_dir, output_dir)

                self._progress_tracker.mark_completed(subject, year)
                self._progress_tracker.save()
                total_processed += 1

        return ExtractionResult.success_result(
            processed=total_processed,
            skipped=total_skipped,
            warnings=warnings,
            metadata={"subjects": len(subject_dirs)},
        )

    def extract(self) -> ExtractionResult:
        """Extract all tests (alias for extract_all).

        Returns:
            ExtractionResult with statistics.
        """
        return self.extract_all()

    def get_progress_tracker(self) -> ProgressTracker:
        """Get the progress tracker instance."""
        return self._progress_tracker

    def clear_progress(self) -> None:
        """Clear all progress tracking."""
        if hasattr(self._progress_tracker, "clear"):
            self._progress_tracker.clear()
