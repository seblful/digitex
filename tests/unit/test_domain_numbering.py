"""The rule that keeps the output tree in order without a renumbering pass.

`numbering_fault` is what replaced the renumbering command: rather than
repairing a tree that went out of order, both sides of the reviewer seam refuse
to write it that way. `preview` is how both sides ask — the replay plus the
fault, with the copy of the book's state taken inside so no caller can forget
it. Both read the disk but need no display, so they are testable here; that the
*two sides agree* is asserted in `tests/characterization`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from digitex.domain.entities import PixelPolygon
from digitex.domain.numbering import numbering_fault, preview
from digitex.domain.placement import (
    PageExtractionState,
    PageRegion,
    PlacedQuestion,
    QuestionPlacement,
)

if TYPE_CHECKING:
    from pathlib import Path

    from digitex.domain.placement import PageLabel

POLYGON = PixelPolygon([(10, 40), (200, 40), (200, 80), (10, 80)])


def _question(joins_next: bool = False) -> PageRegion:
    return PageRegion(label="question", polygon=POLYGON, joins_next=joins_next)


def _marker(label: PageLabel, reading: int | str) -> PageRegion:
    return PageRegion(label=label, polygon=POLYGON, reading=reading)


def _placed(option: int, part: str, *numbers: int) -> list[PlacedQuestion]:
    return [
        PlacedQuestion(
            regions=[PageRegion(label="question", polygon=POLYGON)],
            placement=QuestionPlacement(option=option, part=part, number=number),
        )
        for number in numbers
    ]


def _fill(root: Path, option: int, part: str, *numbers: int) -> None:
    folder = root / str(option) / part
    folder.mkdir(parents=True, exist_ok=True)
    for number in numbers:
        (folder / f"{number}.jpg").write_bytes(b"x")


class TestNumberingFault:
    def test_a_fresh_folder_must_start_at_one(self, tmp_path: Path) -> None:
        assert numbering_fault(_placed(1, "A", 1, 2), tmp_path) is None

        fault = numbering_fault(_placed(1, "A", 2, 3), tmp_path)

        assert fault is not None
        assert (fault.free, fault.placement.number) == (1, 2)
        assert fault.collides is False  # a gap, not an overwrite

    def test_continuing_from_what_is_on_disk_is_clean(self, tmp_path: Path) -> None:
        _fill(tmp_path, 1, "A", 1, 2, 3, 4, 5)

        assert numbering_fault(_placed(1, "A", 6, 7, 8), tmp_path) is None

    def test_landing_on_an_existing_number_is_a_collision(self, tmp_path: Path) -> None:
        _fill(tmp_path, 1, "A", 1, 2, 3, 4, 5)

        fault = numbering_fault(_placed(1, "A", 3, 4), tmp_path)

        assert fault is not None
        assert fault.collides is True
        assert fault.free == 6
        assert fault.position == 0

    def test_landing_past_the_end_would_leave_a_gap(self, tmp_path: Path) -> None:
        """The hole renumbering used to repair, refused before it is written."""
        _fill(tmp_path, 1, "A", 1, 2, 3, 4, 5)

        fault = numbering_fault(_placed(1, "A", 9, 10), tmp_path)

        assert fault is not None
        assert fault.collides is False
        assert fault.free == 6

    def test_only_where_each_folder_starts_is_checked(self, tmp_path: Path) -> None:
        """The rest follow by construction, so they cannot be wrong on their own."""
        _fill(tmp_path, 1, "A", 1)

        assert numbering_fault(_placed(1, "A", 2, 3, 4), tmp_path) is None

    def test_a_reentered_folder_is_checked_like_the_first(self, tmp_path: Path) -> None:
        """A marker mid-page resets the counter, so a re-entered run restarts at 1.

        An option marker OCR missed plus a part marker it read produces exactly
        this shape — the re-entry must not hide behind the folder's first run.
        """
        _fill(tmp_path, 1, "A", 1, 2)

        placed = _placed(1, "A", 3) + _placed(1, "B", 1) + _placed(1, "A", 1)
        fault = numbering_fault(placed, tmp_path)

        assert fault is not None
        assert fault.position == 2
        assert fault.collides is True
        assert fault.free == 3

    def test_a_second_folder_is_checked_on_its_own_terms(self, tmp_path: Path) -> None:
        """A page spanning two parts: the first continues, the second collides."""
        _fill(tmp_path, 1, "A", 1, 2)
        _fill(tmp_path, 1, "B", 1, 2, 3)
        placed = _placed(1, "A", 3, 4) + _placed(1, "B", 2)

        fault = numbering_fault(placed, tmp_path)

        assert fault is not None
        assert fault.placement == QuestionPlacement(1, "B", 2)
        assert fault.position == 2
        assert fault.free == 4

    def test_a_page_with_no_questions_has_nothing_to_fault(
        self, tmp_path: Path
    ) -> None:
        assert numbering_fault([], tmp_path) is None


class TestPreview:
    """The look-without-committing entry both sides of the seam call.

    The copy of the state is taken inside `preview` — the bug class that kills
    is a call site forgetting the copy and silently advancing a whole book.
    """

    def test_the_book_state_is_not_advanced(self, tmp_path: Path) -> None:
        state = PageExtractionState(option=1, part="A", question=4)

        preview([_question(), _question()], state, tmp_path)

        assert (state.option, state.part, state.question) == (1, "A", 4)

    def test_where_the_page_ends_is_reported_on_the_copy(self, tmp_path: Path) -> None:
        state = PageExtractionState(option=1, part="A")

        page = preview([_question(), _question()], state, tmp_path)

        assert (page.ends_at.option, page.ends_at.part, page.ends_at.question) == (
            1,
            "A",
            2,
        )
        assert page.ends_at is not state

    def test_a_clean_page_places_and_reports_no_fault(self, tmp_path: Path) -> None:
        page = preview(
            [_marker("option", 1), _marker("part", "A"), _question(), _question()],
            PageExtractionState(),
            tmp_path,
        )

        assert [str(q.placement) for q in page.placed.questions] == ["1/A/1", "1/A/2"]
        assert page.fault is None
        assert page.continue_helps is False

    def test_a_collision_is_surfaced(self, tmp_path: Path) -> None:
        _fill(tmp_path, 1, "A", 1)

        page = preview([_question()], PageExtractionState(option=1, part="A"), tmp_path)

        assert page.fault is not None
        assert page.fault.collides is True

    def test_moving_the_entry_state_helps_a_fault_before_any_marker(
        self, tmp_path: Path
    ) -> None:
        _fill(tmp_path, 1, "A", 1)

        page = preview([_question()], PageExtractionState(option=1, part="A"), tmp_path)

        assert page.continue_helps is True

    def test_moving_the_entry_state_helps_only_the_entry_group(
        self, tmp_path: Path
    ) -> None:
        """A fault after a marker cannot be moved: the marker sets the counter."""
        _fill(tmp_path, 1, "B", 1)
        regions = [_question(), _marker("part", "B"), _question()]

        page = preview(regions, PageExtractionState(option=1, part="A"), tmp_path)

        assert page.fault is not None
        assert page.continue_helps is False

    def test_joined_pieces_count_once_toward_the_entry_group(
        self, tmp_path: Path
    ) -> None:
        """Two pieces a reviewer joined are one question and take one number."""
        _fill(tmp_path, 1, "B", 1)
        regions = [
            _question(joins_next=True),
            _question(),
            _marker("part", "B"),
            _question(),
        ]

        page = preview(regions, PageExtractionState(option=1, part="A"), tmp_path)

        # The fault is the Part B question, past the one-question entry group.
        assert page.fault is not None
        assert page.fault.position == 1
        assert page.continue_helps is False

    def test_the_placed_regions_are_the_callers_own(self, tmp_path: Path) -> None:
        """A fault names a position; the GUI maps it back to rows by identity."""
        region = _question()

        page = preview([region], PageExtractionState(option=1, part="A"), tmp_path)

        assert page.placed.questions[0].regions[0] is region

    def test_a_question_before_any_marker_raises(self, tmp_path: Path) -> None:
        state = PageExtractionState()

        with pytest.raises(ValueError, match="before any option/part marker"):
            preview([_question()], state, tmp_path)

        assert (state.option, state.part, state.question) == (0, "", 0)
