import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from digitex.core.handlers import PDFHandler
from digitex.core.processors import ImageProcessor

logger = logging.getLogger(__name__)

DEFAULT_MAX_HEIGHT = 2000


def get_device() -> torch.device:
    """Get the best available device for PyTorch operations.

    Returns:
        torch.device: CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        logger.debug("CUDA device available")
        return torch.device("cuda")
    logger.debug("Using CPU device")
    return torch.device("cpu")


def get_device_count() -> int:
    """Get the number of available CUDA devices.

    Returns:
        int: Number of CUDA devices (0 if none available).
    """
    return torch.cuda.device_count()


def get_device_indices() -> list[int]:
    """Get list of available device indices.

    Returns:
        list[int]: List of device indices (empty list if no CUDA devices).
    """
    count = get_device_count()
    return list(range(count))


def prepare_image(
    image: Image.Image,
    max_height: int = DEFAULT_MAX_HEIGHT,
) -> Image.Image:
    """Convert PIL image to BGR, optionally resize, and return as PIL RGB.

    Args:
        image: Input PIL Image.
        max_height: Maximum allowed height. If 0, no resizing.

    Returns:
        Prepared PIL Image in RGB format.
    """
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if max_height > 0:
        processor = ImageProcessor()
        img = processor.resize_image(img, max_height)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


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
