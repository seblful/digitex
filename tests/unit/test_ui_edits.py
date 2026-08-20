"""The page-editing rules, checked without a display.

These are the rules that decide what the reviewer is allowed to approve, and
they used to be reachable only by building a real Tk window — so they were
tested through one, and skipped wherever there was none. `PageEdits` owns them
now, and none of this imports tkinter.

`test_ui_page_review` still covers the half that genuinely needs a widget: that
an edit reaches the tree, that the approve button goes grey, that a crop appears
in the preview pane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from digitex.domain.entities import PixelPolygon
from digitex.domain.placement import PageExtractionState, PageLabel, PageRegion
from digitex.ui.edits import MIN_DRAWN_SIZE, MIN_POINTS, PageEdits

if TYPE_CHECKING:
    from pathlib import Path


def _polygon(
    top: int, left: int = 50, right: int = 550, height: int = 60
) -> PixelPolygon:
    return PixelPolygon(
        [(left, top), (right, top), (right, top + height), (left, top + height)]
    )


def _region(label: PageLabel, top: int, reading: int | str | None = None) -> PageRegion:
    return PageRegion(label=label, polygon=_polygon(top), reading=reading)


def _page() -> list[PageRegion]:
    """One option marker, one part marker, two questions — in reading order."""
    return [
        _region("option", 10, 1),
        _region("part", 90, "A"),
        _region("question", 200),
        _region("question", 300),
    ]


@pytest.fixture
def edits(tmp_path: Path) -> PageEdits:
    made = PageEdits()
    made.load(_page(), PageExtractionState(), tmp_path)
    return made


class TestLoading:
    def test_the_regions_are_copied_not_borrowed(self, tmp_path: Path) -> None:
        """Skipping a page must leave the extractor's own regions as they were."""
        original = _page()
        edits = PageEdits()
        edits.load(original, PageExtractionState(), tmp_path)

        edits.regions[2].label = "part"

        assert original[2].label == "question"

    def test_the_entry_state_is_copied_too(self, tmp_path: Path) -> None:
        state = PageExtractionState(option=2, part="B", question=7)
        edits = PageEdits()
        edits.load(_page(), state, tmp_path)

        edits.set_entry_state(option=5, part="A", question=1)

        assert (state.option, state.part, state.question) == (2, "B", 7)

    def test_loading_clears_the_previous_page_timeline(self, edits: PageEdits) -> None:
        edits.delete(2)
        assert edits.history.can_undo

        edits.load(_page(), PageExtractionState(), edits.output_dir)

        assert edits.history.can_undo is False


class TestNumbering:
    def test_every_question_is_placed_where_it_would_be_saved(
        self, edits: PageEdits
    ) -> None:
        numbering = edits.numbering()

        assert [str(p) for p in numbering.placements] == ["1/A/1", "1/A/2"]
        assert numbering.ok
        assert numbering.ends_at == "page ends at 1/A/2"

    def test_the_pieces_index_into_the_regions(self, edits: PageEdits) -> None:
        """The window colours rows by index, so these have to be region indices."""
        assert sorted(edits.numbering().pieces) == [2, 3]

    def test_a_question_before_any_marker_is_refused(self, tmp_path: Path) -> None:
        edits = PageEdits()
        edits.load([_region("question", 200)], PageExtractionState(), tmp_path)

        numbering = edits.numbering()

        assert numbering.ok is False
        assert "before any option/part marker" in (numbering.problem or "")
        assert numbering.ends_at == ""

    def test_a_number_already_on_disk_is_refused(self, edits: PageEdits) -> None:
        taken = edits.output_dir / "1" / "A"
        taken.mkdir(parents=True)
        (taken / "1.jpg").write_bytes(b"x")

        numbering = edits.numbering()

        assert numbering.ok is False
        assert "already exists" in (numbering.problem or "")

    def test_the_refused_question_is_the_one_marked(self, edits: PageEdits) -> None:
        taken = edits.output_dir / "1" / "A"
        taken.mkdir(parents=True)
        (taken / "1.jpg").write_bytes(b"x")

        # Region 2 is the first question; region 3 numbers fine after it.
        assert edits.numbering().misnumbered == frozenset({2})

    def test_a_gap_is_refused_as_well_as_a_collision(self, edits: PageEdits) -> None:
        """Landing past the end would leave a hole the extractor never fills."""
        taken = edits.output_dir / "1" / "A"
        taken.mkdir(parents=True)
        (taken / "1.jpg").write_bytes(b"x")
        (taken / "2.jpg").write_bytes(b"x")
        edits.set_entry_state(option=1, part="A", question=5)

        numbering = edits.numbering()

        assert numbering.ok is False
        assert "would leave a gap" in (numbering.problem or "")

    def test_numbering_does_not_advance_the_entry_state(self, edits: PageEdits) -> None:
        """Redrawing must not walk the page's starting point forward."""
        edits.numbering()
        edits.numbering()

        assert edits.state.question == 0


class TestWhereThePageStarts:
    def test_a_page_opening_with_a_question_can_be_continued(
        self, tmp_path: Path
    ) -> None:
        edits = PageEdits()
        edits.load([_region("question", 200)], PageExtractionState(), tmp_path)

        assert edits.entry_state_reaches_first_question is True

    def test_a_page_opening_with_a_marker_cannot(self, edits: PageEdits) -> None:
        """The marker sets the counter itself, so moving the entry state is no help."""
        assert edits.entry_state_reaches_first_question is False

    def test_the_remedy_names_continue_only_for_the_entry_group(
        self, tmp_path: Path
    ) -> None:
        """A fault after a marker cannot be fixed by moving the entry state.

        Here the entry group continues cleanly and the Part B group collides —
        recommending 'Continue from disk' would point the reviewer at a button
        that changes nothing.
        """
        taken = tmp_path / "1" / "B"
        taken.mkdir(parents=True)
        (taken / "1.jpg").write_bytes(b"x")

        edits = PageEdits()
        edits.load(
            [
                _region("question", 100),
                _region("part", 200, "B"),
                _region("question", 300),
            ],
            PageExtractionState(option=1, part="A", question=0),
            tmp_path,
        )

        numbering = edits.numbering()

        assert numbering.ok is False
        assert numbering.continue_helps is False
        assert "Continue from disk" not in (numbering.problem or "")

    def test_the_remedy_names_continue_for_a_fault_in_the_entry_group(
        self, tmp_path: Path
    ) -> None:
        taken = tmp_path / "1" / "A"
        taken.mkdir(parents=True)
        (taken / "1.jpg").write_bytes(b"x")

        edits = PageEdits()
        edits.load(
            [_region("question", 100)],
            PageExtractionState(option=1, part="A", question=0),
            tmp_path,
        )

        numbering = edits.numbering()

        assert numbering.continue_helps is True
        assert "Continue from disk" in (numbering.problem or "")

    def test_continue_from_disk_picks_up_where_the_folder_left_off(
        self, tmp_path: Path
    ) -> None:
        taken = tmp_path / "1" / "A"
        taken.mkdir(parents=True)
        for number in (1, 2, 3):
            (taken / f"{number}.jpg").write_bytes(b"x")

        edits = PageEdits()
        edits.load(
            [_region("question", 200)],
            PageExtractionState(option=1, part="A", question=0),
            tmp_path,
        )

        # next_question() hands out question + 1, so 3 makes the next one 4.
        assert edits.continue_from_disk() == 3

    def test_continue_from_disk_is_none_without_a_placement(
        self, tmp_path: Path
    ) -> None:
        edits = PageEdits()
        edits.load([_region("option", 10, 1)], PageExtractionState(), tmp_path)

        assert edits.continue_from_disk() is None


class TestCounts:
    def test_questions_and_markers_are_counted_apart(self, edits: PageEdits) -> None:
        assert (edits.question_count, edits.marker_count) == (2, 2)


class TestJoiningPieces:
    """A question printed in pieces takes one number between them."""

    def test_joining_two_questions_makes_them_one(self, edits: PageEdits) -> None:
        edits.toggle_join_next(2)

        numbering = edits.numbering()
        assert [str(p) for p in numbering.placements] == ["1/A/1"]
        assert str(numbering.pieces[2].placement) == "1/A/1"
        assert str(numbering.pieces[3].placement) == "1/A/1"
        assert (numbering.pieces[3].index, numbering.pieces[3].count) == (2, 2)

    def test_the_last_question_can_wait_for_the_next_page(
        self, edits: PageEdits
    ) -> None:
        edits.toggle_join_next(3)

        numbering = edits.numbering()
        assert [str(p) for p in numbering.placements] == ["1/A/1"]
        assert numbering.pieces[3].held
        assert numbering.held == 1
        # The held piece consumed no number, so the page ends one earlier.
        assert numbering.ends_at == "page ends at 1/A/1"
        assert numbering.ok

    def test_only_a_question_can_be_half_of_one(self, edits: PageEdits) -> None:
        assert edits.toggle_join_next(0) is False
        assert edits.regions[0].joins_next is False

    def test_a_piece_relabelled_a_marker_stops_being_one(
        self, edits: PageEdits
    ) -> None:
        edits.toggle_join_next(2)

        edits.set_label(2, "part")

        assert edits.regions[2].joins_next is False
        assert edits.regions[2].join_offset == (0, 0)

    def test_the_flag_is_undoable(self, edits: PageEdits) -> None:
        edits.toggle_join_next(2)

        assert edits.undo()
        assert edits.regions[2].joins_next is False

    def test_lining_the_pieces_up_is_one_undo_step(self, edits: PageEdits) -> None:
        edits.toggle_join_next(2)

        edits.set_join_offsets({3: (5, -4)})

        assert edits.regions[3].join_offset == (5, -4)
        assert edits.undo()
        assert edits.regions[3].join_offset == (0, 0)
        # Undoing the line-up leaves the join itself alone.
        assert edits.regions[2].joins_next is True

    def test_a_questions_pieces_are_reported_together(self, edits: PageEdits) -> None:
        edits.toggle_join_next(2)

        assert edits.question_pieces(2) == [2, 3]
        assert edits.question_pieces(3) == [2, 3]

    def test_a_whole_question_is_its_own_only_piece(self, edits: PageEdits) -> None:
        assert edits.question_pieces(3) == [3]

    def test_a_marker_is_no_piece_of_anything(self, edits: PageEdits) -> None:
        assert edits.question_pieces(0) == [0]

    def test_a_marker_between_the_pieces_does_not_break_the_join(
        self, edits: PageEdits
    ) -> None:
        """The pieces of a question are its own, whatever is marked up between."""
        edits.toggle_join_next(2)
        edits.reorder(1, 1)  # the part marker, down between the two questions

        assert edits.question_pieces(3) == [1, 3]


class TestCarriedPieces:
    """What a page handed an unfinished piece by the page before it reports."""

    @pytest.fixture
    def carried(self, tmp_path: Path) -> PageEdits:
        made = PageEdits()
        made.load(_page(), PageExtractionState(), tmp_path, carried=1)
        return made

    def test_the_first_question_counts_the_piece_carried_into_it(
        self, carried: PageEdits
    ) -> None:
        pieces = carried.numbering().pieces

        assert (pieces[2].index, pieces[2].count) == (2, 2)
        assert (pieces[3].index, pieces[3].count) == (1, 1)

    def test_a_carried_piece_takes_no_number_of_its_own(
        self, carried: PageEdits
    ) -> None:
        assert [str(p) for p in carried.numbering().placements] == ["1/A/1", "1/A/2"]

    def test_the_carried_piece_belongs_to_the_first_question(
        self, carried: PageEdits
    ) -> None:
        assert carried.first_question == 2
        assert carried.takes_carried(2) is True
        assert carried.takes_carried(3) is False
        assert carried.takes_carried(0) is False

    def test_nothing_carried_means_nothing_to_join(self, edits: PageEdits) -> None:
        assert edits.takes_carried(2) is False

    def test_a_page_that_only_continues_a_question_places_nothing(
        self, carried: PageEdits
    ) -> None:
        carried.toggle_join_next(2)
        carried.delete(3)

        numbering = carried.numbering()
        assert numbering.placements == []
        assert numbering.pieces[2].held
        assert numbering.ok


class TestRelabelling:
    def test_relabelling_a_question_renumbers_the_rest(self, edits: PageEdits) -> None:
        edits.set_label(2, "part")

        # The first question is now a marker, so the second takes its number.
        assert [str(p) for p in edits.numbering().placements] == ["1/A/1"]

    def test_relabelling_drops_a_reading_from_the_old_kind(
        self, edits: PageEdits
    ) -> None:
        """An option number on a part marker would be ignored but still shown."""
        edits.set_label(0, "part")

        assert edits.regions[0].reading is None


class TestDrawing:
    def test_a_box_is_added_in_page_pixels_and_selected(self, edits: PageEdits) -> None:
        added = edits.add_box("question", (100, 200), (300, 400))

        assert added is True
        assert edits.selected == 4
        assert list(edits.regions[4].polygon) == [
            (100, 200),
            (300, 200),
            (300, 400),
            (100, 400),
        ]

    def test_the_corners_may_arrive_in_any_order(self, edits: PageEdits) -> None:
        """Dragging up and to the left is the same box as down and to the right."""
        edits.add_box("question", (300, 400), (100, 200))

        assert next(iter(edits.regions[4].polygon)) == (100, 200)

    def test_a_stray_click_adds_nothing(self, edits: PageEdits) -> None:
        added = edits.add_box("question", (100, 200), (100 + MIN_DRAWN_SIZE - 1, 260))

        assert added is False
        assert len(edits.regions) == 4


class TestPoints:
    def test_a_point_is_inserted_on_the_nearest_edge(self, edits: PageEdits) -> None:
        before = len(list(edits.regions[2].polygon))

        edits.insert_point(2, (300, 200))

        polygon = list(edits.regions[2].polygon)
        assert len(polygon) == before + 1
        # The click sits on the top edge, so it joins between its two corners.
        assert polygon[1] == (300, 200)

    def test_a_polygon_is_not_cut_below_the_croppable_minimum(
        self, edits: PageEdits
    ) -> None:
        assert len(list(edits.regions[2].polygon)) == MIN_POINTS

        assert edits.delete_point(2, 0) is False
        assert len(list(edits.regions[2].polygon)) == MIN_POINTS

    def test_a_point_can_go_once_there_is_one_to_spare(self, edits: PageEdits) -> None:
        edits.insert_point(2, (300, 200))

        assert edits.delete_point(2, 0) is True
        assert len(list(edits.regions[2].polygon)) == MIN_POINTS


class TestMoving:
    def test_nudging_moves_only_that_region(self, edits: PageEdits) -> None:
        untouched = list(edits.regions[3].polygon)

        edits.nudge(2, 10, 0)

        assert next(iter(edits.regions[2].polygon)) == (60, 200)
        assert list(edits.regions[3].polygon) == untouched

    def test_reordering_swaps_with_the_neighbour_and_follows_the_selection(
        self, edits: PageEdits
    ) -> None:
        assert edits.reorder(2, 1) == 3

        assert [r.label for r in edits.regions] == [
            "option",
            "part",
            "question",
            "question",
        ]
        assert edits.selected == 3

    def test_reordering_off_either_end_does_nothing(self, edits: PageEdits) -> None:
        assert edits.reorder(0, -1) is None
        assert edits.reorder(3, 1) is None
        assert edits.history.can_undo is False

    def test_sorting_puts_the_regions_in_reading_order(self, tmp_path: Path) -> None:
        edits = PageEdits()
        edits.load(
            [
                _region("question", 300),
                _region("option", 10, 1),
                _region("part", 90, "A"),
            ],
            PageExtractionState(),
            tmp_path,
        )
        assert edits.numbering().ok is False  # question before its markers

        edits.sort_by_reading_order()

        assert [r.label for r in edits.regions] == ["option", "part", "question"]
        assert edits.numbering().ok

    def test_sorting_keeps_hold_of_the_selected_region(self, edits: PageEdits) -> None:
        """By identity — two regions can carry equal field values."""
        edits.selected = 3
        chosen = edits.regions[3]

        edits.sort_by_reading_order()

        assert edits.selected is not None
        assert edits.regions[edits.selected] is chosen


class TestDeleting:
    def test_deleting_a_marker_makes_the_page_unapprovable(
        self, edits: PageEdits
    ) -> None:
        """Without a marker a crop would land outside {option}/{part}/."""
        edits.delete(0)
        edits.delete(0)

        assert edits.numbering().ok is False

    def test_deleting_clears_the_selection(self, edits: PageEdits) -> None:
        edits.selected = 2

        edits.delete(2)

        assert edits.selected is None


class TestUndo:
    def test_undo_puts_an_edit_back(self, edits: PageEdits) -> None:
        edits.nudge(2, 0, 10)

        assert edits.undo() is True
        assert next(iter(edits.regions[2].polygon)) == (50, 200)

    def test_redo_puts_it_back_again(self, edits: PageEdits) -> None:
        edits.nudge(2, 0, 10)
        edits.undo()

        assert edits.redo() is True
        assert next(iter(edits.regions[2].polygon)) == (50, 210)

    def test_undo_at_the_start_of_the_timeline_reports_nothing_to_do(
        self, edits: PageEdits
    ) -> None:
        assert edits.undo() is False

    def test_undo_restores_the_entry_state_too(self, edits: PageEdits) -> None:
        """An unset entry state is 0/"" — the page's markers supply the rest."""
        edits.set_entry_state(option=4, part="B", question=9)
        edits.commit()

        edits.undo()

        assert (edits.state.option, edits.state.part, edits.state.question) == (
            0,
            "",
            0,
        )

    def test_a_drag_is_one_step_however_far_it_goes(self, edits: PageEdits) -> None:
        """The window commits on button-up, so the moves in between record nothing."""
        edits.drag_polygon(2, 5, 0)
        edits.drag_polygon(2, 5, 0)
        edits.drag_polygon(2, 5, 0)
        edits.commit()

        edits.undo()

        assert next(iter(edits.regions[2].polygon)) == (50, 200)

    def test_dragging_a_vertex_moves_that_point_alone(self, edits: PageEdits) -> None:
        edits.drag_vertex(2, 1, (500, 150))

        assert list(edits.regions[2].polygon) == [
            (50, 200),
            (500, 150),
            (550, 260),
            (50, 260),
        ]
