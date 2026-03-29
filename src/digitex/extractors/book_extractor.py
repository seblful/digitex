"""Book extractor for extracting question images from image files."""

import logging
from pathlib import Path

from tqdm import tqdm

from digitex.extractors.page_extractor import PageExtractor

logger = logging.getLogger(__name__)


class BookExtractor:
    """Extract question images from image files."""

    def __init__(
        self,
        model_path: Path,
        image_format: str,
        question_max_width: int,
        question_max_height: int,
    ) -> None:
        self._page_extractor = PageExtractor(
            model_path=model_path,
            image_format=image_format,
            question_max_width=question_max_width,
            question_max_height=question_max_height,
        )

    def extract(
        self,
        image_dir: Path,
        output_dir: Path,
    ) -> None:
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        from digitex.utils import IMAGE_EXTENSIONS

        images = sorted(
            (p for p in image_dir.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        )

        if not images:
            logger.warning(f"No images found in {image_dir}")
            return

        option_counter = 0
        part_letter = ""
        question_counter = 0

        for image_path in tqdm(images, desc=f"Processing {image_dir.name}", leave=False):
            from PIL import Image

            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            option_counter, part_letter, question_counter = self._page_extractor.extract(
                image, output_dir, option_counter, part_letter, question_counter
            )

        logger.info(f"Extracted images to {output_dir}")
