"""Main extractor that orchestrates book extraction."""

import logging
from pathlib import Path

from tqdm import tqdm

from digitex.config import get_settings
from digitex.extractors.book_extractor import BookExtractor

logger = logging.getLogger(__name__)


class Extractor:
    """Orchestrates extraction of question images from all PDF books."""

    def __init__(
        self,
        model_path: Path | None = None,
        render_scale: int | None = None,
        image_format: str | None = None,
        books_dir: Path | None = None,
        extraction_dir: Path | None = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            model_path: Path to YOLO model. Defaults to settings.extraction.model_path.
            render_scale: PDF render scale factor. Defaults to settings.extraction.render_scale.
            image_format: Output image format. Defaults to settings.extraction.image_format.
            books_dir: Directory containing subject folders. Defaults to settings.extraction.books_dir.
            extraction_dir: Output directory. Defaults to settings.extraction.extraction_dir.
        """
        settings = get_settings()

        self.books_dir = books_dir or settings.extraction.books_dir
        self.extraction_dir = extraction_dir or settings.extraction.extraction_dir

        self.book_extractor = BookExtractor(
            model_path=model_path,
            render_scale=render_scale,
            image_format=image_format,
        )

    def extract_all(self) -> None:
        """Extract question images from all PDFs in the books directory.

        Raises:
            FileNotFoundError: If books directory doesn't exist.
            ValueError: If no detections found on any page.
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

                self.book_extractor.extract(pdf_path, output_dir)
