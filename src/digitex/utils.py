import logging
from pathlib import Path

from PIL import Image

from digitex.core.handlers import PDFHandler
from digitex.core.processors import prepare_image

logger = logging.getLogger(__name__)


def create_pdf_from_images(
    image_dir: str | Path,
    raw_dir: str | Path,
    process: bool = False,
) -> None:
    """Create a PDF from a directory of images.

    Args:
        image_dir: Directory containing the images.
        raw_dir: Directory where the PDF will be saved.
        process: Whether to process images before adding to PDF.

    Raises:
        FileNotFoundError: If image_dir doesn't exist or contains no images.
        IOError: If images cannot be read or PDF cannot be created.
    """
    image_dir = Path(image_dir)
    raw_dir = Path(raw_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    def num_key(x: str) -> int:
        """Extract numeric key from filename for sorting."""
        return int(x.split("_")[-1].split(".")[0])

    image_list = sorted(image_dir.iterdir(), key=lambda p: num_key(p.name))

    if not image_list:
        raise FileNotFoundError(f"No images found in {image_dir}")

    images = []
    for image_path in image_list:
        try:
            image = Image.open(image_path)

            if process:
                image = prepare_image(image)

            images.append(image)
        except (FileNotFoundError, IOError) as e:
            logger.warning(f"Failed to process image {image_path}: {e}")
            continue

    if not images:
        raise ValueError("No valid images could be processed")

    pdf_name = f"{image_dir.name} {raw_dir.name}.pdf"
    pdf_path = raw_dir / pdf_name

    PDFHandler().create_pdf(images, pdf_path)
    logger.info(f"PDF created successfully: {pdf_path}")
