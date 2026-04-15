"""Book extractor for extracting question images from image files."""

from pathlib import Path

from PIL import Image
import structlog
from tqdm import tqdm

from digitex.extractors.page_extractor import PageExtractor
from digitex.utils import _natural_sort_key

logger = structlog.get_logger()


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
            (
                p
                for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=_natural_sort_key,
        )

        if not images:
            logger.warning("No images found", image_dir=image_dir)
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        option_counter = 0
        part_letter = ""
        question_counter = 0

        for image_path in tqdm(
            images, desc=f"Processing {image_dir.name}", leave=False
        ):
            image = Image.open(image_path)

            option_counter, part_letter, question_counter = (
                self._page_extractor.extract(
                    image,
                    output_dir,
                    option_counter,
                    part_letter,
                    question_counter,
                    image_path.name,
                )
            )

        logger.info("Extracted images", output_dir=output_dir)
