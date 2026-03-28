"""Tests extractor that orchestrates extraction of all PDF books."""

import logging
from pathlib import Path

from tqdm import tqdm

from digitex.extractors.book_extractor import BookExtractor

logger = logging.getLogger(__name__)


class TestsExtractor:
    """Orchestrates extraction of question images from all PDF books."""

    def __init__(
        self,
        model_path: Path,
        render_scale: int,
        image_format: str,
        books_dir: Path,
        extraction_dir: Path,
    ) -> None:
        self.books_dir = books_dir
        self.extraction_dir = extraction_dir

        self._book_extractor = BookExtractor(
            model_path=model_path,
            render_scale=render_scale,
            image_format=image_format,
        )

    def extract_all(self) -> None:
        """Extract question images from all PDFs in the books directory.

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
            pdf_dir = subject_dir / "pdf"

            if not pdf_dir.exists():
                logger.warning(f"No pdf folder found in {subject_dir}")
                continue

            pdf_files = list(pdf_dir.glob("*.pdf"))

            if not pdf_files:
                logger.warning(f"No PDF files found in {pdf_dir}")
                continue

            for pdf_path in tqdm(pdf_files, desc=f"Extracting {subject}", leave=False):
                year = pdf_path.stem
                output_dir = self.extraction_dir / subject / year

                self._book_extractor.extract(pdf_path, output_dir)
