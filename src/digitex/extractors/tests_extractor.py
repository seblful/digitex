"""Tests extractor that orchestrates extraction of all image books."""

import logging
from pathlib import Path

from tqdm import tqdm

from digitex.extractors.book_extractor import BookExtractor

logger = logging.getLogger(__name__)


class TestsExtractor:
    """Orchestrates extraction of question images from all image books."""

    def __init__(
        self,
        model_path: Path,
        image_format: str,
        question_max_width: int,
        question_max_height: int,
        books_dir: Path,
        extraction_dir: Path,
    ) -> None:
        self.books_dir = books_dir
        self.extraction_dir = extraction_dir

        self._book_extractor = BookExtractor(
            model_path=model_path,
            image_format=image_format,
            question_max_width=question_max_width,
            question_max_height=question_max_height,
        )

    def extract_all(self) -> None:
        """Extract question images from all subjects in the books directory.

        Raises:
            FileNotFoundError: If books directory doesn't exist.
        """
        if not self.books_dir.exists():
            raise FileNotFoundError(f"Books directory not found: {self.books_dir}")

        subject_dirs = [d for d in self.books_dir.iterdir() if d.is_dir()]

        if not subject_dirs:
            logger.warning(f"No subject folders found in {self.books_dir}")
            return

        for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
            subject = subject_dir.name
            images_dir = subject_dir / "images"

            if not images_dir.exists():
                logger.warning(f"No images folder found in {subject_dir}")
                continue

            year_dirs = [d for d in images_dir.iterdir() if d.is_dir()]

            if not year_dirs:
                logger.warning(f"No year folders found in {images_dir}")
                continue

            for year_dir in tqdm(year_dirs, desc=f"Extracting {subject}", leave=False):
                year = year_dir.name
                output_dir = self.extraction_dir / subject / year
                self._book_extractor.extract(year_dir, output_dir)
