"""Book extractor for extracting question images from image files."""

from pathlib import Path

import structlog
from PIL import Image
from tqdm import tqdm

from digitex.extractors.base import BaseExtractor, ExtractionResult
from digitex.extractors.exceptions import DirectoryNotFoundError
from digitex.extractors.page_extractor import PageExtractor
from digitex.utils import _natural_sort_key

logger = structlog.get_logger()


class BookExtractor(BaseExtractor):
    """Extract question images from a directory of images (a "book")."""

    def __init__(
        self,
        model_path: Path,
        image_format: str = "jpg",
        question_max_width: int = 2000,
        question_max_height: int = 2000,
        page_extractor: PageExtractor | None = None,
    ) -> None:
        """Initialize the book extractor.

        Args:
            model_path: Path to YOLO model file.
            image_format: Output image format.
            question_max_width: Maximum width for extracted questions.
            question_max_height: Maximum height for extracted questions.
            page_extractor: Optional pre-configured PageExtractor (dependency injection).
        """
        self.model_path = model_path
        self.image_format = image_format
        self.question_max_width = question_max_width
        self.question_max_height = question_max_height

        self._page_extractor = page_extractor or PageExtractor(
            model_path=model_path,
            image_format=image_format,
            question_max_width=question_max_width,
            question_max_height=question_max_height,
        )

    def _validate_prerequisites(self) -> None:
        """Validate prerequisites (deferred to extract method)."""
        pass

    def extract(
        self,
        image_dir: Path,
        output_dir: Path,
    ) -> ExtractionResult:
        """Extract question images from a directory of images.

        Args:
            image_dir: Directory containing page images.
            output_dir: Output directory for extracted questions.

        Returns:
            ExtractionResult with statistics.
        """
        from digitex.utils import IMAGE_EXTENSIONS

        if not image_dir.exists():
            raise DirectoryNotFoundError(image_dir)

        images = sorted(
            (
                p
                for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=_natural_sort_key,
        )

        if not images:
            logger.warning("No images found", image_dir=str(image_dir))
            return ExtractionResult.success_result(
                processed=0, warnings=["No images found"]
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        option_counter = 0
        part_letter = ""
        question_counter = 0
        processed_count = 0

        for image_path in tqdm(
            images, desc=f"Processing {image_dir.name}", leave=False
        ):
            try:
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
                processed_count += 1
            except Exception as e:
                logger.error(
                    "Failed to process image",
                    image_path=str(image_path),
                    error=str(e),
                )

        logger.info(
            "Extracted images from book",
            output_dir=str(output_dir),
            processed=processed_count,
        )

        return ExtractionResult.success_result(processed=processed_count)
