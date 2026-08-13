"""Book extractor for extracting question images from image files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from PIL import Image
from tqdm import tqdm

from digitex.domain.corpus import is_image, natural_sort_key
from digitex.pipeline.base import ExtractionResult
from digitex.pipeline.exceptions import DirectoryNotFoundError, ReviewAborted
from digitex.pipeline.page import PageExtractor
from digitex.pipeline.placement import PageExtractionState

if TYPE_CHECKING:
    from pathlib import Path

    from digitex.pipeline.base import ExtractionConfig

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

        Failed page reads are counted in ``failed`` and surfaced as ``errors``
        alongside ``success=True`` — the caller decides whether one bad page
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

        # One state for the whole book: question numbering continues across
        # page boundaries. A page that fails leaves it partly advanced, which
        # is why a book with any error is never marked completed.
        state = PageExtractionState()
        processed_count = 0
        errors: list[str] = []
        warnings: list[str] = []

        for page_number, image_path in enumerate(
            tqdm(images, desc=f"Processing {image_dir.name}", leave=False), start=1
        ):
            try:
                with Image.open(image_path) as image:
                    collisions = self._page_extractor.extract(
                        image,
                        output_dir,
                        state,
                        page_number=page_number,
                        page_count=len(images),
                    )
                processed_count += 1
                # Not a page failure — resuming an unfinished year replays its
                # pages over their own output — but the caller must see it, or
                # a diverged numbering silently loses crops.
                warnings.extend(
                    f"{image_path.name}: {placement} already extracted,"
                    " kept the existing image"
                    for placement in collisions
                )
            except ReviewAborted:
                # Not a page failure: the reviewer stopped the run. Let it out
                # so no caller counts this book as finished.
                raise
            except Exception as e:
                msg = f"Failed to process {image_path.name}: {e}"
                logger.error(
                    "Failed to process page",
                    image_path=str(image_path),
                    error=str(e),
                    exc_info=True,
                )
                errors.append(msg)

        logger.info(
            "Extracted images from book",
            output_dir=str(output_dir),
            processed=processed_count,
            failed=len(errors),
        )

        return ExtractionResult(
            success=True,  # partial success — caller can inspect errors
            processed=processed_count,
            failed=len(errors),
            errors=errors,
            warnings=warnings,
        )
