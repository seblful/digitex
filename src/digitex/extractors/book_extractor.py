"""Book extractor for extracting question images from a single PDF book."""

import logging
from pathlib import Path

import pypdfium2 as pdfium
from tqdm import tqdm

from digitex.core.handlers import PDFHandler
from digitex.extractors.page_extractor import PageExtractor

logger = logging.getLogger(__name__)


class BookExtractor:
    """Extract question images from a single PDF book."""

    def __init__(
        self,
        model_path: Path,
        render_scale: int,
        image_format: str,
    ) -> None:
        self._page_extractor = PageExtractor(
            model_path=model_path,
            render_scale=render_scale,
            image_format=image_format,
        )
        self._pdf_handler = PDFHandler()
        self.render_scale = render_scale

    def extract(
        self,
        pdf_path: Path,
        output_dir: Path,
    ) -> None:
        """Extract question images from a single PDF file.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Output directory for extracted images.

        Raises:
            FileNotFoundError: If PDF file doesn't exist.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pdf = pdfium.PdfDocument(str(pdf_path))
        num_pages = len(pdf)

        option_counter = 0
        part_letter = ""
        question_counter = 0

        for page_idx in tqdm(
            range(num_pages), desc=f"Processing {pdf_path.name}", leave=False
        ):
            page = pdf[page_idx]
            image = self._pdf_handler.get_page_image(page, scale=self.render_scale)

            option_counter, part_letter, question_counter = self._page_extractor.extract(
                image, output_dir, option_counter, part_letter, question_counter
            )

        pdf.close()
        logger.info(f"Extracted images to {output_dir}")
