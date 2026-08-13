"""Question numbering — from a page's labelled regions to where each one lands.

:func:`place_questions` is the single walk from regions to placements, shared
by the preview the review GUI draws and the write :class:`PageExtractor`
performs. The two differ only in the writer they pass, so what a reviewer
approves is what lands on disk.

Pure: no PIL, no YOLO, no filesystem. Reading regions off a page image belongs
to PageExtractor, and drawing them to the GUI.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import structlog

if TYPE_CHECKING:
    from digitex.core.domain import PixelPolygon

logger = structlog.get_logger()

# A conflict-resolver correction moves the question to a different option, and
# an option always starts at Part A. Named once so the state machine and the
# path it lands at cannot disagree.
CORRECTED_PART = "A"

PageLabel = Literal["option", "part", "question"]
"""What the segmentation model can find on a page."""


@dataclass(frozen=True)
class QuestionPlacement:
    """Where one detected question lands in the extraction output."""

    option: int
    part: str
    number: int

    def __str__(self) -> str:
        return f"{self.option}/{self.part}/{self.number}"


@dataclass
class PageExtractionState:
    """Question-numbering state machine, threaded across a book's pages.

    Owns every decision about which option/part/number a detection belongs
    to. Consumes the page's markers in reading order (``on_option`` /
    ``on_part``), hands out placements as values (``next_question`` +
    ``commit_question``), and takes conflict-resolver corrections back via
    ``correct_option``. Performs no I/O — reading markers off the page and
    saving crops belong to PageExtractor.
    """

    option: int = 0
    part: str = ""
    question: int = 0

    def on_option(self, new_option: int | None) -> bool:
        """Advance when a marker continues the option sequence.

        Anything that is not exactly the next option number is treated as an
        OCR misread and ignored. Returns True on change.
        """
        if new_option is not None and new_option == self.option + 1:
            self.option = new_option
            self.part = "A"
            self.question = 0
            return True
        return False

    def on_part(self, new_part: str | None) -> bool:
        """Switch part when a different part marker is read. Returns True on change."""
        if new_part is not None and new_part != self.part:
            self.part = new_part
            self.question = 0
            return True
        return False

    def next_question(self) -> QuestionPlacement:
        """Return the placement the next question will get, without committing.

        The caller commits via :meth:`commit_question` only after the crop is
        saved, so a failed save doesn't consume a question number.
        """
        return QuestionPlacement(self.option, self.part, self.question + 1)

    def commit_question(self) -> None:
        """Consume the question number handed out by :meth:`next_question`."""
        self.question += 1

    def correct_option(self, resolved_option: int) -> bool:
        """Apply a conflict-resolver decision. Returns True if the option moved.

        The question counter deliberately keeps running — the corrected
        question retains its number under the new option.
        """
        if resolved_option == self.option:
            return False
        self.option = resolved_option
        self.part = CORRECTED_PART
        return True

    def adopt(self, other: PageExtractionState) -> None:
        """Take *other*'s position — how a reviewer corrects where a page starts.

        The book's state is one object threaded across every page, so a
        reviewer that hands back a different entry point moves this one rather
        than replacing it.
        """
        self.option = other.option
        self.part = other.part
        self.question = other.question


@dataclass
class PageRegion:
    """One labelled region on a page, with whatever was read off it.

    Mutable, because the review GUI edits polygons, labels and readings in
    place. ``reading`` is the option number or part letter a marker carries —
    None when OCR could not read it, and always None for a question.
    """

    label: PageLabel
    polygon: PixelPolygon
    reading: int | str | None = None


@dataclass(frozen=True)
class PlacedQuestion:
    """A question region and the option/part/number it was handed."""

    region: PageRegion
    placement: QuestionPlacement


QuestionWriter = Callable[[PageRegion, QuestionPlacement], int]
"""Persist one question's crop; returns the option it actually landed under.

A writer that returns anything other than ``placement.option`` has moved the
question, and every later question on the page follows it — that is how the
conflict resolver's correction reaches the numbering.
"""


def place_only(region: PageRegion, placement: QuestionPlacement) -> int:  # noqa: ARG001
    """Writer that writes nothing — the preview the review GUI draws."""
    return placement.option


def reading_order_key(polygon: PixelPolygon) -> tuple[int, int]:
    """Sort key for reading order: top to bottom, then left to right."""
    return (min(p[1] for p in polygon), min(p[0] for p in polygon))


def place_questions(
    regions: Iterable[PageRegion],
    state: PageExtractionState,
    write: QuestionWriter = place_only,
) -> list[PlacedQuestion]:
    """Replay *regions* through *state*, handing each question its placement.

    *state* is mutated: markers advance it, questions consume numbers from it,
    and a writer reporting a different option corrects it. Pass a copy to
    preview a page without committing to it.

    Args:
        regions: The page's labelled regions, in reading order.
        state: Where the book's numbering stands on entering the page.
        write: What to do with each placed question. Writes nothing by default.

    Returns:
        One entry per question region, in the order they were placed.

    Raises:
        ValueError: If a question comes before any option/part marker was read.
    """
    placed: list[PlacedQuestion] = []

    for region in regions:
        if region.label == "option":
            reading = region.reading if isinstance(region.reading, int) else None
            if state.on_option(reading):
                logger.debug("Option changed", option_counter=state.option)
        elif region.label == "part":
            reading = region.reading if isinstance(region.reading, str) else None
            if state.on_part(reading):
                logger.debug("Part changed", part_letter=state.part)
        elif region.label == "question":
            placement = state.next_question()
            if not placement.option or not placement.part:
                # pathlib drops an empty segment, so this would land one
                # directory short of {option}/{part}/ and be invisible to
                # every reader of the output tree.
                raise ValueError(
                    "Question detected before any option/part marker was read"
                )

            resolved = write(region, placement)
            state.commit_question()
            if state.correct_option(resolved):
                logger.info(
                    "Option corrected",
                    from_option=placement.option,
                    to_option=resolved,
                )
            placed.append(PlacedQuestion(region=region, placement=placement))

    return placed
