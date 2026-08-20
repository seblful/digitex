"""Question numbering — from a page's labelled regions to where each one lands.

:func:`place_questions` is the single walk from regions to placements, shared by
the preview the review GUI draws and the write :class:`PageExtractor` performs.
The two differ only in the writer they pass, which is what makes what a reviewer
approves the same thing that lands on disk.

A question can be printed in two pieces, one on each side of a page break. The
walk groups the regions a reviewer joined into one question, and hands back the
pieces left over at the end of a page for the next page to finish — see
:mod:`digitex.pipeline.pieces`.

Pure: no PIL, no YOLO, no filesystem. Reading regions off a page image belongs to
PageExtractor, and drawing them to the GUI.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

import structlog

from digitex.domain.entities import PixelPolygon

logger = structlog.get_logger()

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

    Owns every decision about which option/part/number a detection belongs to.
    It consumes a page's markers in reading order (:meth:`on_option`,
    :meth:`on_part`), hands out placements as values (:meth:`next_question` then
    :meth:`commit_question`), and takes a reviewer's correction through
    :meth:`adopt`. It performs no I/O: reading markers off the page and saving
    crops belong to PageExtractor.
    """

    option: int = 0
    part: str = ""
    question: int = 0

    def on_option(self, new_option: int | None) -> bool:
        """Advance when a marker continues the option sequence.

        Anything that is not exactly the next option number is treated as an OCR
        misread and ignored — a page's markers are the only evidence, and a
        wrong jump would re-file every question after it. Returns True on change.
        """
        if new_option is None or new_option != self.option + 1:
            return False
        self.option = new_option
        self.part = "A"
        self.question = 0
        return True

    def on_part(self, new_part: str | None) -> bool:
        """Switch part when a different part marker is read.

        Returns True on change.
        """
        if new_part is None or new_part == self.part:
            return False
        self.part = new_part
        self.question = 0
        return True

    def next_question(self) -> QuestionPlacement:
        """The placement the next question will get, without committing to it.

        The caller commits through :meth:`commit_question` only once the crop is
        saved, so a failed save does not consume a question number.
        """
        return QuestionPlacement(self.option, self.part, self.question + 1)

    def commit_question(self) -> None:
        """Consume the question number handed out by :meth:`next_question`."""
        self.question += 1

    def adopt(self, other: PageExtractionState) -> None:
        """Take *other*'s position — how a reviewer corrects where a page starts.

        The book's state is one object threaded across every page, so a reviewer
        handing back a different entry point moves this one rather than replacing
        it.
        """
        self.option = other.option
        self.part = other.part
        self.question = other.question


@dataclass
class PageRegion:
    """One labelled region on a page, with whatever was read off it.

    Mutable, because the review GUI edits polygons, labels and readings in place.
    ``reading`` is the option number or part letter a marker carries — None when
    OCR could not read it, and always None for a question.

    ``joins_next`` marks a question that is only a piece of one: its image
    continues into the next question region, either further down this page or at
    the top of the page after it. A piece is not a question of its own, so it
    takes no number and is saved as part of the question it joins.
    ``join_offset`` is how this piece sits against the piece above it when it is
    not the first — pixels right and down from where it would otherwise land,
    which is how the two halves are lined up into one readable question. Only the
    review GUI sets either: nothing about a page says where a question was cut in
    two.
    """

    label: PageLabel
    polygon: PixelPolygon
    reading: int | str | None = None
    joins_next: bool = False
    join_offset: tuple[int, int] = (0, 0)


def copy_regions(regions: Iterable[PageRegion]) -> list[PageRegion]:
    """Copy regions deeply enough that editing one cannot reach another."""
    return [
        PageRegion(
            label=region.label,
            polygon=PixelPolygon(list(region.polygon)),
            reading=region.reading,
            joins_next=region.joins_next,
            join_offset=region.join_offset,
        )
        for region in regions
    ]


@dataclass(frozen=True)
class PlacedQuestion:
    """One question's regions and the option/part/number they were handed.

    More than one region when the question was printed in pieces and the reviewer
    joined them: their crops stack into the single image the placement names.
    """

    regions: list[PageRegion]
    placement: QuestionPlacement


@dataclass(frozen=True)
class PagePlacement:
    """Where a page's questions land, and what it leaves the next page to finish.

    ``held`` is the run of pieces at the end of the page whose question is not
    finished on it. Nothing is written for them and they take no number, so a
    page ending mid-question costs the numbering nothing — the question is
    numbered on the page that completes it. The caller carries their crops across
    (:class:`digitex.pipeline.pieces.PageCarry`).
    """

    questions: list[PlacedQuestion]
    held: list[PageRegion] = field(default_factory=list)


QuestionWriter = Callable[[list[PageRegion], QuestionPlacement], None]
"""Persist one placed question's crop, stacked from its pieces in reading order."""


def write_nothing(regions: list[PageRegion], placement: QuestionPlacement) -> None:
    """Writer that writes nothing — the preview the review GUI draws."""


def reading_order_key(polygon: PixelPolygon) -> tuple[int, int]:
    """Sort key for reading order: top to bottom, then left to right."""
    return (min(p[1] for p in polygon), min(p[0] for p in polygon))


def place_questions(
    regions: Iterable[PageRegion],
    state: PageExtractionState,
    write: QuestionWriter = write_nothing,
) -> PagePlacement:
    """Replay *regions* through *state*, handing each question its placement.

    *state* is mutated: markers advance it and questions consume numbers from it.
    Pass a copy to preview a page without committing to it.

    A question region marked ``joins_next`` is a piece of the question that
    follows it: it is collected rather than placed, and the next question is
    written as those pieces plus itself. Pieces still collected at the end of the
    page come back as ``held``, for the caller to hand on to the page that
    finishes them.

    Args:
        regions: The page's labelled regions, in reading order.
        state: Where the book's numbering stands on entering the page.
        write: What to do with each placed question. Writes nothing by default.

    Returns:
        The questions placed on this page, in order, and the trailing pieces the
        next page has to finish.

    Raises:
        ValueError: If a question comes before any option/part marker was read.
    """
    placed: list[PlacedQuestion] = []
    pieces: list[PageRegion] = []

    for region in regions:
        if region.label == "option":
            # A misread marker reads as text, not a number; ignore it rather
            # than let it advance the sequence.
            option = region.reading if isinstance(region.reading, int) else None
            if state.on_option(option):
                logger.debug("Option changed", option_counter=state.option)

        elif region.label == "part":
            part = region.reading if isinstance(region.reading, str) else None
            if state.on_part(part):
                logger.debug("Part changed", part_letter=state.part)

        elif region.label == "question":
            pieces.append(region)
            if region.joins_next:
                # Not a question of its own: it takes no number, and its crop is
                # written with the question it joins — which the next page may be
                # the one to hold.
                continue

            placement = state.next_question()
            if not placement.option or not placement.part:
                # pathlib drops an empty segment, so this would land one
                # directory short of {option}/{part}/ and be invisible to every
                # reader of the output tree.
                raise ValueError(
                    "Question detected before any option/part marker was read"
                )

            write(pieces, placement)
            state.commit_question()
            placed.append(PlacedQuestion(regions=pieces, placement=placement))
            pieces = []

    return PagePlacement(questions=placed, held=pieces)
