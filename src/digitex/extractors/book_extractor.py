"""Book extractor for extracting question images from image files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from PIL import Image
from tqdm import tqdm

from digitex.core.corpus import is_image, natural_sort_key
from digitex.extractors.base import ExtractionResult
from digitex.extractors.exceptions import DirectoryNotFoundError
from digitex.extractors.page_extractor import PageExtractionState, PageExtractor

if TYPE_CHECKING:
    from pathlib import Path

    from digitex.extractors.base import ExtractionConfig

logger = structlog.get_logger()


class BookExtractor:
    """Extract question images from a directory of images (a "book").

    ``page_extractor`` is the one injectable collaborator. A caller that needs
    a custom conflict resolver builds a configured :class:`PageExtractor` and
    passes it here, rather than threading the resolver down two more levels.
    """

    def __init__(
        self,
        config: ExtractionConfig,
        page_extractor: PageExtractor | None = None,
    ) -> None:
        self._page_extractor = page_extractor or PageExtractor(config)

    def extract(
        self,
        image_dir: Path,
        output_dir: Path,
    ) -> ExtractionResult:
        """Extract question images from a directory of images.

        Failed page reads are counted as ``failed`` in the result metadata
        and surfaced as errors — the caller can decide whether one bad page
        invalidates the whole book.
        """
        if not image_dir.exists():
            raise DirectoryNotFoundError(image_dir)

        images = sorted(
            (p for p in image_dir.iterdir() if is_image(p)),
            key=natural_sort_key,
        )

        if not images:
            logger.warning("No images found", image_dir=str(image_dir))
            return ExtractionResult.success_result(
                processed=0, warnings=["No images found"]
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        state = PageExtractionState()
        processed_count = 0
        errors: list[str] = []

        for image_path in tqdm(
            images, desc=f"Processing {image_dir.name}", leave=False
        ):
            try:
                image = Image.open(image_path)
                state = self._page_extractor.extract(
                    image, output_dir, state, image_path.name
                )
                processed_count += 1
            except Exception as e:
                msg = f"Failed to process {image_path.name}: {e}"
                logger.error(
                    "Failed to process page",
                    image_path=str(image_path),
                    error=str(e),
                )
                errors.append(msg)

        logger.info(
            "Extracted images from book",
            output_dir=str(output_dir),
            processed=processed_count,
            failed=len(errors),
        )

        if errors:
            return ExtractionResult(
                success=True,  # partial success — caller can inspect errors
                processed=processed_count,
                errors=errors,
                metadata={"failed": len(errors)},
            )
        return ExtractionResult.success_result(processed=processed_count)
