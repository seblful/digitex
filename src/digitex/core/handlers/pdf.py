"""PDF handling utilities."""

import logging
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_RENDER_SCALE = 3


class PDFHandler:
    """Handler for PDF file operations and page rendering."""

    @staticmethod
    def create_pdf(images: list[Image.Image], output_path: str | Path) -> None:
        """Create a PDF document from a list of PIL images.

        Args:
            images: List of PIL Images to convert to PDF pages.
            output_path: Path where the PDF will be saved.

        Raises:
            IOError: If the PDF cannot be created or saved.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pdf = pdfium.PdfDocument.new()

        for image in images:
            bitmap = pdfium.PdfBitmap.from_pil(image)
            pdf_image = pdfium.PdfImage.new(pdf)
            pdf_image.set_bitmap(bitmap)

            width, height = pdf_image.get_size()
            matrix = pdfium.PdfMatrix().scale(width, height)
            pdf_image.set_matrix(matrix)

            page = pdf.new_page(width, height)
            page.insert_obj(pdf_image)
            page.gen_content()

            bitmap.close()

        pdf.save(str(output_path), version=17)

    def get_page_image(
        self,
        page: pdfium.PdfPage,
        scale: int = DEFAULT_RENDER_SCALE,
    ) -> Image.Image:
        """Render a PDF page to a PIL Image.

        Args:
            page: PDF page object to render.
            scale: Rendering scale factor (higher = better quality).

        Returns:
            PIL Image of the rendered page in RGB format.
        """
        bitmap = page.render(scale=scale, rotation=0)
        image = bitmap.to_pil()
        return image if image.mode == "RGB" else image.convert("RGB")

    def get_random_image(
        self,
        pdf_listdir: list[str],
        pdf_dir: str | Path,
    ) -> tuple[Image.Image, str, int]:
        """Get a random page from a random PDF in the directory.

        Args:
            pdf_listdir: List of PDF filenames.
            pdf_dir: Directory containing the PDF files.

        Returns:
            Tuple of (image, image_name, page_index).

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If the PDF cannot be read.
        """
        import random

        pdf_dir = Path(pdf_dir)

        rand_pdf_name = random.choice(pdf_listdir)
        rand_pdf_path = pdf_dir / rand_pdf_name

        if not rand_pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {rand_pdf_path}")

        rand_pdf_obj = pdfium.PdfDocument(str(rand_pdf_path))

        rand_page_idx = random.randint(0, len(rand_pdf_obj) - 1)
        rand_page = rand_pdf_obj[rand_page_idx]

        rand_image = self.get_page_image(page=rand_page)
        rand_image_name = rand_pdf_path.stem + ".jpg"

        rand_pdf_obj.close()

        return rand_image, rand_image_name, rand_page_idx
