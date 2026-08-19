"""Checking a page's regions by hand before anything is cropped.

A `PageReviewer` is a callable that, given what the extractor is about to do to
one page, returns the version to actually carry out — or None to skip the page
and write nothing. The default accepts the proposal untouched, so extraction
without a reviewer behaves exactly as it did before there was one.

The shape mirrors `conflict_resolution`: a type alias rather than a Protocol,
because a callable is the smallest thing that expresses "given a page, approve
it". The one interactive reviewer lives in `digitex.ui.page_review`; nothing
in this package imports it, so the pipeline stays free of any GUI toolkit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from digitex.domain.corpus import highest_question_number

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

    from digitex.domain.entities import PixelPolygon
    from digitex.pipeline.pieces import HeldPiece
    from digitex.pipeline.placement import (
        PageExtractionState,
        PageRegion,
        PlacedQuestion,
        QuestionPlacement,
    )


QuestionCrop = Callable[[Sequence["PageRegion"], Sequence["HeldPiece"]], "Image.Image"]
"""Cut one question out of the page exactly as saving it would.

A question is its pieces stacked, so it comes as a sequence of regions — one
for a whole question, more for one printed across a page break, each carrying
how it is lined up against the piece above it. The second argument is the
pieces cut from an earlier page: already cut, so they are handed over as images
rather than as polygons of a page that is no longer open.
"""

PieceCrop = Callable[["PixelPolygon"], "Image.Image"]
"""Cut one piece of a question out of the page, at the page's own scale.

What a reviewer lines two pieces up against each other on. The saved file is
the pieces stacked and then capped to the question size, so the cap is no part
of this.
"""


@dataclass(frozen=True)
class PageProposal:
    """What the extractor is about to write for one page, before it writes it.

    ``regions`` and ``state`` are the reviewer's own copies: edit them, hand
    them back, or drop them — the extractor's originals move only when it
    adopts a returned :class:`ReviewedPage`. ``output_dir`` is the year
    directory the crops land in, which is also what a reviewer counts to show
    its progress.

    ``crop`` is the extractor's own cropping pipeline, bound to this page. A
    reviewer showing a question's crop shows the file that would be written,
    not its own idea of one — the same reason the numbering preview replays
    :func:`place_questions` rather than reimplementing it. ``crop_piece`` is the
    same pipeline stopped one step earlier, at a single piece before the pieces
    are stacked, for lining two of them up. Both are None when the extractor
    was not asked for them.

    ``page_number`` and ``page_count`` are the page's place in its book, for a
    reviewer that wants to say how far along the run is. Both are 0 when the
    caller extracts a page on its own, outside a book.

    ``carried`` holds the pieces an earlier page could not finish, in reading
    order. They belong to this page's first question, whose image is saved as
    them stacked on top of it — a reviewer must show that, because the file
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
    a piece behind that this page does not continue.
    """

    regions: list[PageRegion]
    state: PageExtractionState
    discard_carried: bool = False


PageReviewer = Callable[[PageProposal], "ReviewedPage | None"]
"""Approve a page as-is, hand back a corrected one, or return None to skip it."""


def accept_page(proposal: PageProposal) -> ReviewedPage:
    """Default reviewer: take the proposal as it stands, no interaction."""
    return ReviewedPage(regions=proposal.regions, state=proposal.state)


@dataclass(frozen=True)
class NumberingFault:
    """A question whose number would not continue its option/part folder."""

    position: int
    """Index into the placed questions, so a caller can point at the region."""

    placement: QuestionPlacement
    free: int
    """The number the folder's next image must carry."""

    @property
    def collides(self) -> bool:
        """True when the slot is taken, False when a gap would be left."""
        return self.placement.number < self.free


def numbering_fault(
    placed: Sequence[PlacedQuestion], output_dir: Path
) -> NumberingFault | None:
    """The first placement that does not continue what *output_dir* already holds.

    A question's file has to be the next one in its option/part folder. Landing
    on a number already there would overwrite an extracted question; landing
    past the end leaves the folder with a hole. Catching both before the write
    is what keeps the output tree in order without a renumbering pass after it.

    Only where each folder run starts is checked — within a run the numbers
    follow by construction. A page that re-enters a folder (a marker read
    mid-page resets the counter) starts a new run, checked like the first.
    """
    previous: tuple[int, str] | None = None

    for position, question in enumerate(placed):
        placement = question.placement
        folder = (placement.option, placement.part)
        if folder == previous:
            continue
        previous = folder

        free = highest_question_number(output_dir, placement.option, placement.part) + 1
        if placement.number != free:
            return NumberingFault(position=position, placement=placement, free=free)

    return None
