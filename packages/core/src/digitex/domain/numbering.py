"""Whether a page's numbers continue the output tree, or break it.

The one rule both sides of the reviewer seam consult: the review window will not
approve a page that fails it, and the extractor replays every page through it
before writing. A gap refuses the page — there is no renumbering pass to close a
hole afterwards — while a collision is survivable, because a year resumed after
an interruption meets its own earlier output on every page it replays.

Numbering rather than reviewing, which is why it does not live in
:mod:`digitex.pipeline.review`: an unattended run applies exactly the same rule
with no reviewer anywhere in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from digitex.domain.corpus import highest_question_number

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from digitex.domain.placement import PlacedQuestion, QuestionPlacement


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
    on a number already there would overwrite an extracted question; landing past
    the end leaves the folder with a hole. Catching both before the write is what
    keeps the output tree in order without a renumbering pass after it.

    Only the start of each folder run is checked — within a run the numbers
    follow by construction. A page that re-enters a folder (a marker read
    mid-page resets the counter) starts a new run, checked like the first.
    """
    checked: tuple[int, str] | None = None

    for position, question in enumerate(placed):
        placement = question.placement
        folder = (placement.option, placement.part)
        if folder == checked:
            continue
        checked = folder

        free = highest_question_number(output_dir, placement.option, placement.part) + 1
        if placement.number != free:
            return NumberingFault(position=position, placement=placement, free=free)

    return None
