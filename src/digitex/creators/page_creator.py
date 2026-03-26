"""Page data creator for extracting random pages from PDFs."""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from digitex.core.handlers.pdf import PDFHandler
from digitex.core.processors.image import ImageProcessor

logger = logging.getLogger(__name__)


class PageDataCreator:
    """Creator for extracting and processing random pages from PDFs."""

    def __init__(self, scale: int = 3) -> None:
        """Initialize the PageDataCreator with required handlers.

        Args:
            scale: PDF rendering scale factor (higher = better quality).
        """
        self.scale = scale
        self.image_processor = ImageProcessor()
        self.pdf_handler = PDFHandler()

    def _process_image(self, image: Image.Image) -> Image.Image:
        """Remove blue color from an image.

        Args:
            image: Input PIL Image.

        Returns:
            Image with blue color removed.
        """
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img = self.image_processor.remove_color(img)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _save_image(
        self,
        page_index: int,
        output_dir: Path,
        image: Image.Image,
        image_name: str,
        num_saved: int,
        total_images: int,
    ) -> int:
        """Save an image to the output directory.

        Args:
            page_index: Index of the page in the PDF.
            output_dir: Directory where the image will be saved.
            image: PIL Image to save.
            image_name: Original name of the image.
            num_saved: Current count of saved images.
            total_images: Total number of images to save.

        Returns:
            Updated count of saved images.
        """
        image_stem = Path(image_name).stem
        image_path = output_dir / f"{image_stem}_{page_index}.jpg"

        if not image_path.exists():
            image.save(image_path, "JPEG")
            num_saved += 1
            logger.info(f"{num_saved}/{total_images} images saved.")

        return num_saved

    def create(
        self,
        pdf_dir: str | Path,
        output_dir: str | Path,
        num_images: int,
    ) -> None:
        """Extract and save random pages from PDFs.

        Args:
            pdf_dir: Directory containing PDF files.
            output_dir: Directory where images will be saved.
            num_images: Number of images to extract.

        Raises:
            FileNotFoundError: If pdf_dir does not exist.
            IOError: If images cannot be saved.
        """
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_listdir = [pdf for pdf in os.listdir(pdf_dir) if pdf.endswith(".pdf")]
        num_saved = 0

        logger.info(f"Extracting {num_images} images from PDFs in {pdf_dir}")

        while num_images > num_saved:
            rand_image, rand_image_name, rand_page_idx = (
                self.pdf_handler.get_random_image(
                    pdf_listdir=pdf_listdir,
                    pdf_dir=pdf_dir,
                    scale=self.scale,
                )
            )
            rand_image = self._process_image(rand_image)
            num_saved = self._save_image(
                page_index=rand_page_idx,
                output_dir=output_dir,
                image=rand_image,
                image_name=rand_image_name,
                num_saved=num_saved,
                total_images=num_images,
            )

        logger.info(f"Successfully extracted {num_images} images to {output_dir}")
