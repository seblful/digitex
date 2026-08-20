"""One year of one subject: every page in a book directory, in reading order.

The middle of the three runners. It owns the two things a book's pages share and
two books never do — the numbering state and the carried question pieces — and
it owns the decision that a bad page is worth one report rather than the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from PIL import Image
from tqdm import tqdm

from digitex.domain.corpus import is_image, natural_sort_key
from digitex.domain.placement import PageExtractionState
from digitex.pipeline.exceptions import DirectoryNotFoundError, ReviewAborted
from digitex.pipeline.outcome import (
    BookReport,
    Collision,
    PageFailure,
    UnfinishedPieces,
)
from digitex.pipeline.pieces import PageCarry

if TYPE_CHECKING:
    from pathlib import Path

    from digitex.pipeline.page import PageExtractor

logger = structlog.get_logger()


class BookExtractor:
    """Extract question images from a directory of images (a "book").

    ``page_extractor`` does every page. A caller that needs a reviewer or a
    different detector configures the :class:`PageExtractor` and passes it in
    rather than threading those knobs down through here — which is also why no
    extraction config reaches this class.
    """

    def __init__(self, page_extractor: PageExtractor) -> None:
        self._page_extractor = page_extractor

    def extract(
        self,
        image_dir: Path,
        output_dir: Path,
    ) -> BookReport:
        """Extract question images from a directory of images.

        A page that raises is recorded as a failure and the run carries on — the
        caller decides whether one bad page invalidates the whole book.
        Collisions are recorded too, and are not failures: an interrupted year
        meets its own earlier output on every page it replays.

        Raises:
            DirectoryNotFoundError: If *image_dir* does not exist.
            ReviewAborted: If the reviewer stopped the run. Deliberately not
                caught: no caller may count this book as finished.
        """
        if not image_dir.exists():
            raise DirectoryNotFoundError(image_dir)

        pages = sorted(
            (path for path in image_dir.iterdir() if is_image(path)),
            key=natural_sort_key,
        )

        if not pages:
            logger.warning("No images found", image_dir=str(image_dir))
            return BookReport(note="No images found")

        output_dir.mkdir(parents=True, exist_ok=True)

        # One state for the whole book: question numbering continues across page
        # boundaries. A page that fails leaves it partly advanced, which is why a
        # book with any error is never marked completed.
        state = PageExtractionState()
        # One carry too: a question printed across a page break is written by the
        # page that finishes it, out of the piece the page before it left here.
        # Per book, because no question spans two years.
        carry = PageCarry()

        processed = 0
        collisions: list[Collision] = []
        failures: list[PageFailure] = []

        for page_number, page_path in enumerate(
            tqdm(pages, desc=f"Processing {image_dir.name}", leave=False), start=1
        ):
            try:
                with Image.open(page_path) as image:
                    taken = self._page_extractor.extract(
                        image,
                        output_dir,
                        state,
                        page_number=page_number,
                        page_count=len(pages),
                        carry=carry,
                    )
                processed += 1
                # Not a page failure — resuming an unfinished year replays its
                # pages over their own output — but the caller must see it, or a
                # diverged numbering silently loses crops.
                collisions.extend(
                    Collision(page=page_path.name, placement=placement)
                    for placement in taken
                )
            except ReviewAborted:
                # The reviewer stopped the run rather than this page failing.
                raise
            except Exception as e:
                failures.append(PageFailure(page=page_path.name, cause=str(e)))
                logger.error(
                    "Failed to process page",
                    image_path=str(page_path),
                    error=str(e),
                    exc_info=True,
                )

        # A piece still held at the end of the book was never joined to
        # anything, so no file carries it — which the caller has to hear.
        unfinished = tuple(
            UnfinishedPieces(page=piece.page_name, count=1) for piece in carry.take()
        )

        logger.info(
            "Extracted images from book",
            output_dir=str(output_dir),
            processed=processed,
            failed=len(failures),
        )

        return BookReport(
            pages=processed,
            collisions=tuple(collisions),
            failures=tuple(failures),
            unfinished=unfinished,
        )
