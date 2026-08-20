"""The working copy of one page, and every rule about editing it.

Nothing here imports tkinter. The review window is a view over this object: it
draws what :meth:`PageEdits.numbering` reports and turns clicks and keystrokes
into the operations below. Which is what makes the rules that actually matter
assertable — what relabelling does to a misread marker, when a number is
refused, where "continue from disk" would put the counter, what counts as a
stray click. None of that could be reached while it lived inside a widget.

The division of labour with the window: an operation that changes the page
records its own undo step, and the window redraws afterwards. A live drag is the
one exception — :meth:`drag_polygon` and :meth:`drag_vertex` deliberately record
nothing, so a drag across the page lands in the timeline once, when the button
comes up.

A question printed across a page break is two regions of one question. The
reviewer marks the first with :meth:`toggle_join_next`, lines the pieces up with
:meth:`set_join_offsets`, and :meth:`numbering` reports which question each
region is a piece of.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from digitex.domain.corpus import highest_question_number
from digitex.domain.entities import PixelPolygon
from digitex.domain.numbering import numbering_fault
from digitex.domain.placement import (
    PageExtractionState,
    PageRegion,
    QuestionPlacement,
    copy_regions,
    place_questions,
    reading_order_key,
)
from digitex.ui import geometry
from digitex.ui.history import EditHistory

if TYPE_CHECKING:
    from digitex.domain.numbering import NumberingFault
    from digitex.domain.placement import PageLabel, PagePlacement
    from digitex.ui.history import EditSnapshot

# A crop needs four points — cut_out_image_by_polygon refuses fewer.
MIN_POINTS = 4

# A drawn box smaller than this, in page pixels, is a stray click.
MIN_DRAWN_SIZE = 5


@dataclass(frozen=True)
class QuestionPiece:
    """Where one question region lands: which question, and which piece of it.

    ``placement`` is None while the piece waits for the next page to finish the
    question it belongs to — nothing is written for it here, and it took no
    number. ``index`` is its 1-based place among the question's pieces, counting
    any carried onto this page from an earlier one, and ``count`` how many pieces
    the question has — 0 while the page cannot know, because the rest of it is on
    the next page.
    """

    placement: QuestionPlacement | None = None
    index: int = 1
    count: int = 1

    @property
    def held(self) -> bool:
        """True when the next page is the one that finishes this question."""
        return self.placement is None

    @property
    def alone(self) -> bool:
        """True when this region is a whole question rather than a piece of one."""
        return self.count == 1


@dataclass(frozen=True)
class Numbering:
    """What replaying the current regions through the entry state produces.

    ``pieces`` says, for every question region, which question it lands in and
    which piece of it it is — one whole question per region on nearly every page,
    and more than one region per question where a reviewer joined the halves of a
    question printed across a page break.

    ``problem`` is what stops the page being approved: either the placement walk
    refused it outright, or one question's number would collide with the output
    tree or leave a hole in it. ``misnumbered`` indexes into the regions list, so
    the offending row and polygon can be drawn in red. ``continue_helps`` says
    the fault sits in the entry group, which is the only case where moving where
    the page starts — the 'Continue from disk' button — can fix anything.
    """

    pieces: dict[int, QuestionPiece] = field(default_factory=dict)
    misnumbered: frozenset[int] = frozenset()
    problem: str | None = None
    continue_helps: bool = False
    ends_at: str = ""

    @property
    def ok(self) -> bool:
        return self.problem is None

    @property
    def placements(self) -> list[QuestionPlacement]:
        """Every question this page writes, in order — one entry per file."""
        placements: list[QuestionPlacement] = []
        for piece in self.pieces.values():
            if piece.placement is not None and piece.placement not in placements:
                placements.append(piece.placement)
        return placements

    @property
    def held(self) -> int:
        """How many pieces this page hands to the next one."""
        return sum(1 for piece in self.pieces.values() if piece.held)


class PageEdits:
    """One page's regions and entry state, with an undo timeline over them.

    Holds a copy of both: approving hands these back to the extractor, skipping
    drops them, and the proposal's own objects are never touched either way.
    """

    def __init__(self) -> None:
        self.regions: list[PageRegion] = []
        self.state = PageExtractionState()
        self.selected: int | None = None
        self.output_dir = Path()
        self.carried = 0
        self.history = EditHistory()

    # --- loading ---

    def load(
        self,
        regions: list[PageRegion],
        state: PageExtractionState,
        output_dir: Path,
        carried: int = 0,
    ) -> None:
        """Take on a new page, discarding the previous one's timeline.

        *carried* is how many question pieces the page before this one left
        unfinished. They belong to this page's first question, so what that
        question is saved as is not the crop of this page alone.
        """
        self.regions = copy_regions(regions)
        self.state = replace(state)
        self.output_dir = output_dir
        self.carried = carried
        self.selected = None
        self.history.reset(self.regions, self.state)

    # --- what the window reports ---

    def numbering(self) -> Numbering:
        """Replay the regions through a copy of the entry state.

        A copy, because the real entry state describes where the page *starts* —
        advancing it here would walk the preview a page further along every time
        anything was redrawn.
        """
        preview = replace(self.state)

        try:
            placed = place_questions(self.regions, preview)
        except ValueError as exc:
            return Numbering(problem=str(exc))

        # By identity: two regions can hold equal field values, and it is the
        # position in the list the window colours a row by.
        at = {id(region): index for index, region in enumerate(self.regions)}
        misnumbered, problem, continue_helps = self._fault(placed, at)

        return Numbering(
            pieces=self._pieces(placed, at),
            misnumbered=misnumbered,
            problem=problem,
            continue_helps=continue_helps,
            ends_at=f"page ends at {preview.option}/{preview.part or '?'}/"
            f"{preview.question}",
        )

    def _pieces(
        self, placed: PagePlacement, at: dict[int, int]
    ) -> dict[int, QuestionPiece]:
        """Which question each region landed in, keyed by its index in the list."""
        pieces: dict[int, QuestionPiece] = {}
        # Whatever was carried onto the page belongs to its first question,
        # whichever that turns out to be — including one the next page still has
        # to finish, which is a question printed across three pages.
        carried = self.carried

        for question in placed.questions:
            for offset, region in enumerate(question.regions):
                pieces[at[id(region)]] = QuestionPiece(
                    placement=question.placement,
                    index=offset + 1 + carried,
                    count=len(question.regions) + carried,
                )
            carried = 0

        for offset, region in enumerate(placed.held):
            pieces[at[id(region)]] = QuestionPiece(index=offset + 1 + carried, count=0)
        return pieces

    def _fault(
        self, placed: PagePlacement, at: dict[int, int]
    ) -> tuple[frozenset[int], str | None, bool]:
        """Whether the page's numbers continue the output tree, and what to say."""
        fault = numbering_fault(placed.questions, self.output_dir)
        if fault is None:
            return frozenset(), None, False

        offender = placed.questions[fault.position]
        # Only a fault the entry state is still numbering can be moved by
        # changing where the page starts.
        continue_helps = fault.position < self._entry_group_size
        return (
            frozenset(at[id(region)] for region in offender.regions),
            self._fault_message(fault, continue_helps=continue_helps),
            continue_helps,
        )

    @staticmethod
    def _fault_message(fault: NumberingFault, *, continue_helps: bool) -> str:
        """Say what is wrong with a number, and which remedy applies."""
        wrong = "already exists" if fault.collides else "would leave a gap"
        remedy = (
            "Use 'Continue from disk'."
            if continue_helps
            else "A marker starts this group, so the page's own numbering is"
            " right — skip the page if it is already extracted, or correct the"
            " marker above it."
        )
        return (
            f"{fault.placement} {wrong} — the next free number in"
            f" {fault.placement.option}/{fault.placement.part} is {fault.free}."
            f" {remedy}"
        )

    @property
    def _entry_group_size(self) -> int:
        """Questions before the first marker — the group the entry state numbers.

        Questions, not regions: two pieces a reviewer joined are one question and
        take one number between them.
        """
        count = 0
        joined = False
        for region in self.regions:
            if region.label in ("option", "part"):
                break
            if region.label == "question":
                if not joined:
                    count += 1
                joined = region.joins_next
        return count

    @property
    def entry_state_reaches_first_question(self) -> bool:
        """True when nothing resets the numbering before the first question.

        An option or part marker sets the counter itself, so moving where the
        page starts cannot move a group that begins after one.
        """
        return self._entry_group_size > 0

    def continue_from_disk(self) -> int | None:
        """The entry counter that puts the first question in the free slot.

        None when there is no placement to continue from. ``next_question()``
        hands out ``question + 1``, so the counter sits one below the free
        number.
        """
        first = next(iter(self.numbering().placements), None)
        if first is None:
            return None
        free = highest_question_number(self.output_dir, first.option, first.part) + 1
        return max(free - 1, 0)

    def question_pieces(self, index: int) -> list[int]:
        """Every region index making up the question *index* is a piece of.

        Just *index* for a whole question, and for anything that is not a
        question at all — a marker is nobody's piece. Markers between two pieces
        are skipped rather than breaking the run: what a reviewer joined stays
        joined however the page is marked up between the halves.
        """
        if self.regions[index].label != "question":
            return [index]

        group: list[int] = []
        for at, region in enumerate(self.regions):
            if region.label != "question":
                continue
            group.append(at)
            if region.joins_next:
                continue
            if index in group:
                return group
            group = []
        # What is left is the run the next page finishes, and *index* is a
        # question, so it is either in a closed group above or in this one.
        return group

    @property
    def first_question(self) -> int | None:
        """The first question region on the page, or None if it has none."""
        return next(
            (
                at
                for at, region in enumerate(self.regions)
                if region.label == "question"
            ),
            None,
        )

    def takes_carried(self, index: int) -> bool:
        """True when the pieces carried onto this page join *index*'s question."""
        if not self.carried or self.regions[index].label != "question":
            return False
        return self.question_pieces(index)[0] == self.first_question

    @property
    def question_count(self) -> int:
        return sum(1 for region in self.regions if region.label == "question")

    @property
    def marker_count(self) -> int:
        return len(self.regions) - self.question_count

    # --- the undo timeline ---

    def commit(self) -> None:
        """Record the current page in the timeline."""
        self.history.push(self.regions, self.state, self.selected)

    def undo(self) -> bool:
        return self._restore(self.history.undo())

    def redo(self) -> bool:
        return self._restore(self.history.redo())

    def _restore(self, snapshot: EditSnapshot | None) -> bool:
        """Adopt a snapshot. False at either end of the timeline."""
        if snapshot is None:
            return False
        # EditHistory hands back copies, so these are ours to edit in place.
        self.regions = snapshot.regions
        self.state = snapshot.state
        self.selected = snapshot.selected
        return True

    # --- edits, each recording an undo step ---

    def add_box(
        self, label: PageLabel, corner: tuple[int, int], opposite: tuple[int, int]
    ) -> bool:
        """Add a rectangular region, selecting it. False for a stray click.

        The two corners may arrive in either order — dragging up and to the left
        is the same box as down and to the right. A rectangle is enough shape to
        draw: the cropper reduces any polygon to its minimum-area quad before
        warping, and masks with the polygon itself.
        """
        left, right = sorted((corner[0], opposite[0]))
        top, bottom = sorted((corner[1], opposite[1]))
        if right - left < MIN_DRAWN_SIZE or bottom - top < MIN_DRAWN_SIZE:
            return False

        self.regions.append(
            PageRegion(
                label=label,
                polygon=PixelPolygon(
                    [(left, top), (right, top), (right, bottom), (left, bottom)]
                ),
            )
        )
        self.selected = len(self.regions) - 1
        self.commit()
        return True

    def insert_point(self, index: int, point: tuple[int, int]) -> None:
        """Add a vertex on the polygon edge nearest *point*."""
        polygon = list(self.regions[index].polygon)
        polygon.insert(geometry.nearest_edge(polygon, point) + 1, point)
        self.regions[index].polygon = PixelPolygon(polygon)
        self.commit()

    def delete_point(self, index: int, point: int) -> bool:
        """Remove a vertex. False when the polygon is already at ``MIN_POINTS``."""
        polygon = list(self.regions[index].polygon)
        if len(polygon) <= MIN_POINTS:
            return False
        del polygon[point]
        self.regions[index].polygon = PixelPolygon(polygon)
        self.commit()
        return True

    def set_label(self, index: int, label: PageLabel) -> None:
        region = self.regions[index]
        region.label = label
        # The old reading belonged to the old kind — an option number on a part
        # marker would be ignored by the walk and still shown as if it counted.
        region.reading = None
        if label != "question":
            # Only a question can be a piece of a question.
            region.joins_next = False
            region.join_offset = (0, 0)
        self.commit()

    def toggle_join_next(self, index: int) -> bool:
        """Mark a question as continuing into the next piece, or stop marking it.

        False when the region is not a question — the only thing that can be half
        of one.
        """
        region = self.regions[index]
        if region.label != "question":
            return False
        region.joins_next = not region.joins_next
        self.commit()
        return True

    def set_join_offsets(self, offsets: dict[int, tuple[int, int]]) -> None:
        """Line the pieces of one question up, in a single undo step.

        Keyed by region index; each offset is where that piece sits against the
        piece above it.
        """
        for index, offset in offsets.items():
            self.regions[index].join_offset = offset
        self.commit()

    def set_reading(self, index: int, reading: int | str | None) -> None:
        self.regions[index].reading = reading
        self.commit()

    def delete(self, index: int) -> None:
        del self.regions[index]
        # Whatever was selected either went or moved down one, and neither is
        # worth guessing at.
        self.selected = None
        self.commit()

    def reorder(self, index: int, delta: int) -> int | None:
        """Swap a region with its neighbour in the reading order.

        Returns the region's new index, or None when it is already at that end
        and nothing moved.
        """
        target = index + delta
        if not 0 <= target < len(self.regions):
            return None
        regions = self.regions
        regions[index], regions[target] = regions[target], regions[index]
        self.selected = target
        self.commit()
        return target

    def sort_by_reading_order(self) -> None:
        """Put the regions in the order the numbering walk follows."""
        chosen = self.regions[self.selected] if self.selected is not None else None
        self.regions.sort(key=lambda region: reading_order_key(region.polygon))
        # Found by identity, not equality: two regions can hold equal fields.
        self.selected = next(
            (at for at, region in enumerate(self.regions) if region is chosen), None
        )
        self.commit()

    def nudge(self, index: int, dx: int, dy: int) -> None:
        self._move_polygon(index, dx, dy)
        self.commit()

    def set_entry_state(self, option: int, part: str, question: int) -> None:
        """Set where the page starts numbering. Records nothing.

        Typing in a spinbox should reach the timeline once it settles, and that
        debounce is the window's to arrange — one step per keystroke would fill
        the timeline with digits.
        """
        self.state.option = option
        self.state.part = part
        self.state.question = question

    # --- live drag: mutate now, record on release ---

    def drag_polygon(self, index: int, dx: int, dy: int) -> None:
        self._move_polygon(index, dx, dy)

    def drag_vertex(self, index: int, point: int, to: tuple[int, int]) -> None:
        polygon = list(self.regions[index].polygon)
        polygon[point] = to
        self.regions[index].polygon = PixelPolygon(polygon)

    def _move_polygon(self, index: int, dx: int, dy: int) -> None:
        region = self.regions[index]
        region.polygon = PixelPolygon(geometry.moved(list(region.polygon), dx, dy))
