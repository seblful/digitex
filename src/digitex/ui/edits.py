"""The working copy of one page, and every rule about editing it.

Nothing here imports tkinter. The review window is a view over this object: it
draws what :meth:`PageEdits.numbering` reports and turns clicks and keystrokes
into the operations below. Which means the rules that actually matter — what
relabelling does to a misread reading, when a number is refused, where
"continue from disk" would put the counter, what counts as a stray click — can
be asserted directly, and could not while they lived inside a widget.

The split of responsibility is: operations that change the page record their own
undo step, and the window redraws afterwards. A live drag is the exception —
:meth:`drag_polygon` and :meth:`drag_vertex` deliberately record nothing, so a
drag across the page lands in the timeline once, when the button comes up.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from digitex.domain.corpus import highest_question_number
from digitex.domain.entities import PixelPolygon
from digitex.pipeline.placement import (
    PageExtractionState,
    PageRegion,
    place_questions,
    reading_order_key,
)
from digitex.pipeline.review import numbering_fault
from digitex.ui import geometry
from digitex.ui.history import EditHistory, copy_regions

if TYPE_CHECKING:
    from digitex.pipeline.placement import PageLabel, QuestionPlacement
    from digitex.pipeline.review import NumberingFault
    from digitex.ui.history import EditSnapshot

# A crop needs four points — ImageCropper refuses fewer.
MIN_POINTS = 4

# A drawn box smaller than this, in page pixels, is a stray click.
MIN_DRAWN_SIZE = 5


@dataclass(frozen=True)
class Numbering:
    """What replaying the current regions through the entry state produces.

    ``problem`` is what stops the page being approved: either the placement
    walk refused it outright, or one question's number would collide with the
    output tree or leave a hole in it. ``misnumbered`` indexes into the regions
    list so the offending row and polygon can be drawn in red.
    """

    placements: dict[int, QuestionPlacement] = field(default_factory=dict)
    misnumbered: frozenset[int] = frozenset()
    problem: str | None = None
    ends_at: str = ""

    @property
    def ok(self) -> bool:
        return self.problem is None


class PageEdits:
    """One page's regions and entry state, with an undo timeline over them.

    Holds a copy: approving hands these back to the extractor, skipping drops
    them, and the proposal's own objects are never touched.
    """

    def __init__(self) -> None:
        self.regions: list[PageRegion] = []
        self.state = PageExtractionState()
        self.selected: int | None = None
        self.output_dir = Path()
        self.history = EditHistory()

    # --- loading ---

    def load(
        self,
        regions: list[PageRegion],
        state: PageExtractionState,
        output_dir: Path,
    ) -> None:
        """Take on a new page, discarding the previous one's timeline."""
        self.regions = copy_regions(regions)
        self.state = replace(state)
        self.output_dir = output_dir
        self.selected = None
        self.history.reset(self.regions, self.state)

    # --- what the window reports ---

    def numbering(self) -> Numbering:
        """Replay the regions through a copy of the entry state.

        A copy, because the real entry state describes where the page *starts*
        — advancing it here would make the preview drift a page further along
        every time anything was redrawn.
        """
        preview = replace(self.state)

        try:
            placed = place_questions(self.regions, preview)
        except ValueError as exc:
            return Numbering(problem=str(exc))

        questions = [i for i, r in enumerate(self.regions) if r.label == "question"]
        placements = {
            index: item.placement for index, item in zip(questions, placed, strict=True)
        }

        misnumbered: frozenset[int] = frozenset()
        problem: str | None = None
        fault = numbering_fault(placed, self.output_dir)
        if fault is not None:
            misnumbered = frozenset({questions[fault.position]})
            problem = self._fault_message(fault)

        part = preview.part or "?"
        return Numbering(
            placements=placements,
            misnumbered=misnumbered,
            problem=problem,
            ends_at=f"page ends at {preview.option}/{part}/{preview.question}",
        )

    def _fault_message(self, fault: NumberingFault) -> str:
        """Say what is wrong with a number, and which remedy applies."""
        wrong = "already exists" if fault.collides else "would leave a gap"
        remedy = (
            "Use 'Continue from disk'."
            if self.entry_state_reaches_first_question
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
    def entry_state_reaches_first_question(self) -> bool:
        """True when nothing resets the numbering before the first question.

        An option or part marker sets the counter itself, so moving where the
        page starts cannot move a group that begins after one.
        """
        for region in self.regions:
            if region.label == "question":
                return True
            if region.label in ("option", "part"):
                return False
        return False

    def continue_from_disk(self) -> int | None:
        """The entry counter that puts the first question in the free slot.

        None when there is no placement to continue from. ``next_question()``
        hands out ``question + 1``, so the counter sits one below the free
        number.
        """
        first = next(iter(self.numbering().placements.values()), None)
        if first is None:
            return None
        free = highest_question_number(self.output_dir, first.option, first.part) + 1
        return max(free - 1, 0)

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
        if snapshot is None:
            return False
        # EditHistory hands back copies, so these are ours to edit.
        self.regions = snapshot.regions
        self.state = snapshot.state
        self.selected = snapshot.selected
        return True

    # --- edits, each recording an undo step ---

    def add_box(
        self, label: PageLabel, corner: tuple[int, int], opposite: tuple[int, int]
    ) -> bool:
        """Add a rectangular region, selecting it. False for a stray click.

        A rectangle is enough: the cropper reduces any polygon to its
        minimum-area quad before warping, and masks with the polygon itself.
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
        # The old reading belongs to the old kind — an option number on a part
        # marker would be ignored anyway, and shown as if it counted.
        region.reading = None
        self.commit()

    def set_reading(self, index: int, reading: int | str | None) -> None:
        self.regions[index].reading = reading
        self.commit()

    def delete(self, index: int) -> None:
        del self.regions[index]
        self.selected = None
        self.commit()

    def reorder(self, index: int, delta: int) -> int | None:
        """Swap a region with its neighbour in the reading order. New index, or None."""
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
        selected = self.regions[self.selected] if self.selected is not None else None
        self.regions.sort(key=lambda region: reading_order_key(region.polygon))
        # By identity, not equality: two regions can hold equal field values.
        self.selected = next(
            (i for i, r in enumerate(self.regions) if r is selected), None
        )
        self.commit()

    def nudge(self, index: int, dx: int, dy: int) -> None:
        self._move_polygon(index, dx, dy)
        self.commit()

    def set_entry_state(self, option: int, part: str, question: int) -> None:
        """Set where the page starts numbering. Records nothing.

        Typing in a spinbox should reach the timeline once it settles, which is
        the window's debounce to arrange, not one step per keystroke.
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
