"""Checking a page's regions by hand before anything is cropped.

A ``PageReviewer`` is a callable: given what the extractor is about to do to one
page, it returns the version to actually carry out — or None to skip the page
and write nothing. The default accepts the proposal untouched, so extraction
without a reviewer behaves exactly as it did before there was one.

A type alias rather than a Protocol class, because a callable is the smallest
thing that expresses "given a page, approve it". The one interactive reviewer
lives in :mod:`digitex.ui.page_review`; nothing here imports it, which is what
keeps the pipeline free of any GUI toolkit.

Which *numbers* a page may be approved onto is not here. That is numbering, not
reviewing, and lives in :mod:`digitex.domain.numbering` where both sides of
this seam reach it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

    from digitex.domain.entities import PixelPolygon
    from digitex.domain.placement import PageExtractionState, PageRegion
    from digitex.pipeline.pieces import HeldPiece


QuestionCrop = Callable[[Sequence["PageRegion"], Sequence["HeldPiece"]], "Image.Image"]
"""Cut one question out of the page exactly as saving it would.

A question is its pieces stacked, so it takes a sequence of regions — one for a
whole question, more for one printed across a page break, each carrying how it
lines up against the piece above it. The second argument is the pieces cut from
an earlier page: already cut, so they arrive as images rather than as polygons
of a page that is no longer open.
"""

PieceCrop = Callable[["PixelPolygon"], "Image.Image"]
"""Cut one piece of a question out of the page, at the page's own scale.

What a reviewer lines two pieces up against each other on. The saved file is
the pieces stacked and *then* capped to the question size, so the cap is no
part of this.
"""


@dataclass(frozen=True)
class PageProposal:
    """What the extractor is about to write for one page, before it writes it.

    ``regions`` and ``state`` are the reviewer's own copies: edit them, hand
    them back, or drop them — the extractor's originals move only when it adopts
    a returned :class:`ReviewedPage`. ``output_dir`` is the year directory the
    crops land in, which is also what a reviewer counts to show its progress.

    ``crop`` is the extractor's own cropping pipeline, bound to this page, so a
    reviewer previewing a question shows the file that *would be written* rather
    than its own likeness of one — the same reason the numbering preview replays
    :func:`place_questions` instead of reimplementing it. ``crop_piece`` is that
    pipeline stopped one step earlier, at a single piece before the pieces are
    stacked, for lining two of them up. Both are None when the extractor was not
    asked for them.

    ``page_number`` and ``page_count`` place the page in its book, for a
    reviewer that wants to say how far along the run is. Both are 0 when a
    caller extracts a page on its own, outside a book.

    ``carried`` holds the pieces an earlier page could not finish, in reading
    order. They belong to this page's first question, whose image is saved as
    them stacked on top of it — a reviewer has to show that, because the file
    approving writes is not the crop of this page alone.
    """

    image: Image.Image
    regions: list[PageRegion]
    state: PageExtractionState
    output_dir: Path
    page_name: str = ""
    crop: QuestionCrop | None = None
    crop_piece: PieceCrop | None = None
    page_number: int = 0
    page_count: int = 0
    carried: list[HeldPiece] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewedPage:
    """A reviewer's verdict: what to write, and where to start numbering it.

    ``discard_carried`` throws away the pieces handed to this page instead of
    joining them to its first question — the way out when the page before left
    behind a piece this page does not continue.
    """

    regions: list[PageRegion]
    state: PageExtractionState
    discard_carried: bool = False


PageReviewer = Callable[[PageProposal], "ReviewedPage | None"]
"""Approve a page as-is, hand back a corrected one, or return None to skip it."""


def accept_page(proposal: PageProposal) -> ReviewedPage:
    """Default reviewer: take the proposal as it stands, no interaction."""
    return ReviewedPage(regions=proposal.regions, state=proposal.state)
