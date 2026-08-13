"""Checking a page's regions by hand before anything is cropped.

A `PageReviewer` is a callable that, given what the extractor is about to do to
one page, returns the version to actually carry out — or None to skip the page
and write nothing. The default accepts the proposal untouched, so extraction
without a reviewer behaves exactly as it did before there was one.

The shape mirrors `conflict_resolution`: a type alias rather than a Protocol,
because a callable is the smallest thing that expresses "given a page, approve
it". The one interactive reviewer lives in `digitex.gui.page_review`; nothing
in this package imports it, so the extractors stay free of any GUI toolkit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from digitex.domain.corpus import highest_question_number

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

    from digitex.domain.entities import PixelPolygon
    from digitex.extractors.placement import (
        PageExtractionState,
        PageRegion,
        PlacedQuestion,
        QuestionPlacement,
    )


QuestionCrop = Callable[["PixelPolygon"], "Image.Image"]
"""Cut one region out of the page exactly as saving it would."""


@dataclass(frozen=True)
class PageProposal:
    """What the extractor is about to write for one page, before it writes it.

    ``regions`` and ``state`` are the extractor's own objects: a reviewer that
    edits them in place and approves gets exactly that written. One that means
    to skip should leave them alone. ``output_dir`` is the year directory the
    crops land in, which is also what a reviewer counts to show its progress.

    ``crop`` is the extractor's own cropping pipeline, bound to this page. A
    reviewer showing a question's crop shows the file that would be written,
    not its own idea of one — the same reason the numbering preview replays
    :func:`place_questions` rather than reimplementing it. None when the
    extractor was not asked for one.

    ``page_number`` and ``page_count`` are the page's place in its book, for a
    reviewer that wants to say how far along the run is. Both are 0 when the
    caller extracts a page on its own, outside a book.
    """

    image: Image.Image
    regions: list[PageRegion]
    state: PageExtractionState
    output_dir: Path
    page_name: str = ""
    crop: QuestionCrop | None = None
    page_number: int = 0
    page_count: int = 0


@dataclass(frozen=True)
class ReviewedPage:
    """A reviewer's verdict: what to write, and where to start numbering it."""

    regions: list[PageRegion]
    state: PageExtractionState


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

    Only where each folder's run starts is checked — the questions after it
    follow by construction.
    """
    started: set[tuple[int, str]] = set()

    for position, question in enumerate(placed):
        placement = question.placement
        folder = (placement.option, placement.part)
        if folder in started:
            continue
        started.add(folder)

        free = highest_question_number(output_dir, placement.option, placement.part) + 1
        if placement.number != free:
            return NumberingFault(position=position, placement=placement, free=free)

    return None
