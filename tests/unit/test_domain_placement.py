"""Tests for the question-numbering state machine and the placement walk.

Both are pure, so nothing here touches a page image, a model or the disk: the
walk's only contact with the outside world is the writer it is handed, and
these tests hand it a recorder.
"""

from dataclasses import replace

import pytest

from digitex.domain.entities import PixelPolygon
from digitex.domain.placement import (
    PageExtractionState,
    PageLabel,
    PageRegion,
    QuestionPlacement,
    copy_regions,
    place_questions,
    reading_order_key,
)

POLYGON = PixelPolygon([(10, 40), (200, 40), (200, 80), (10, 80)])


def _question() -> PageRegion:
    return PageRegion(label="question", polygon=POLYGON)


def _marker(label: PageLabel, reading: int | str | None) -> PageRegion:
    return PageRegion(label=label, polygon=POLYGON, reading=reading)


class _Writer:
    """Records what it was asked to write, in the order it was asked."""

    def __init__(self) -> None:
        self.written: list[QuestionPlacement] = []
        self.pieces: list[int] = []

    def __call__(self, regions: list[PageRegion], placement: QuestionPlacement) -> None:
        self.written.append(placement)
        self.pieces.append(len(regions))


class TestPageExtractionState:
    """The question-numbering state machine through its interface."""

    def test_option_marker_advances_sequentially(self) -> None:
        state = PageExtractionState()
        assert state.on_option(1) is True
        assert (state.option, state.part, state.question) == (1, "A", 0)

    def test_non_sequential_option_marker_ignored(self) -> None:
        state = PageExtractionState(option=1, part="B", question=3)
        assert state.on_option(5) is False
        assert state.on_option(None) is False
        assert (state.option, state.part, state.question) == (1, "B", 3)

    def test_part_marker_switches_and_resets_numbering(self) -> None:
        state = PageExtractionState(option=1, part="A", question=7)
        assert state.on_part("B") is True
        assert (state.part, state.question) == ("B", 0)

    def test_same_or_missing_part_marker_ignored(self) -> None:
        state = PageExtractionState(option=1, part="A", question=7)
        assert state.on_part("A") is False
        assert state.on_part(None) is False
        assert state.question == 7

    def test_adopt_moves_the_book_state_to_another_position(self) -> None:
        """A reviewer corrects where a page starts by handing back its own state."""
        state = PageExtractionState(option=1, part="A", question=3)
        state.adopt(PageExtractionState(option=4, part="B", question=7))
        assert (state.option, state.part, state.question) == (4, "B", 7)

    def test_placement_renders_as_its_output_path(self) -> None:
        assert str(QuestionPlacement(option=3, part="A", number=5)) == "3/A/5"


class TestPlaceQuestions:
    def test_markers_place_the_questions_that_follow_them(self) -> None:
        writer = _Writer()
        state = PageExtractionState()

        placed = place_questions(
            [_marker("option", 1), _marker("part", "A"), _question(), _question()],
            state,
            write=writer,
        )

        assert [p.placement for p in placed.questions] == [
            QuestionPlacement(1, "A", 1),
            QuestionPlacement(1, "A", 2),
        ]
        assert writer.written == [p.placement for p in placed.questions]
        assert (state.option, state.part, state.question) == (1, "A", 2)

    def test_a_reading_of_the_wrong_type_counts_as_unreadable(self) -> None:
        """The GUI writes readings by hand, so the walk narrows rather than trusts."""
        state = PageExtractionState(option=2, part="A")

        place_questions(
            [_marker("option", "3"), _marker("part", 7), _question()], state
        )

        assert (state.option, state.part) == (2, "A")

    def test_the_default_writer_writes_nothing_but_still_numbers(self) -> None:
        """This is the preview the review GUI draws."""
        state = PageExtractionState(option=1, part="A", question=4)

        placed = place_questions([_question(), _question()], state)

        assert [p.placement.number for p in placed.questions] == [5, 6]

    def test_previewing_a_copy_leaves_the_book_state_alone(self) -> None:
        state = PageExtractionState(option=1, part="A", question=4)

        place_questions([_question()], replace(state))

        assert state.question == 4

    def test_a_failed_write_does_not_consume_the_number(self) -> None:
        """A retried page hands the same number out again, leaving no hole."""
        state = PageExtractionState(option=1, part="A")

        def failing_writer(
            regions: list[PageRegion], placement: QuestionPlacement
        ) -> None:
            raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            place_questions([_question()], state, write=failing_writer)

        assert state.question == 0

    def test_question_before_any_marker_raises_without_writing(self) -> None:
        writer = _Writer()

        with pytest.raises(ValueError, match="before any option/part marker"):
            place_questions([_question()], PageExtractionState(), write=writer)

        assert writer.written == []

    def test_regions_are_placed_in_the_order_given_not_by_position(self) -> None:
        """Reordering in the review window is what fixes a two-column page."""
        lower = PageRegion(
            label="part",
            polygon=PixelPolygon([(10, 900), (99, 900), (99, 950), (10, 950)]),
        )
        lower.reading = "B"
        state = PageExtractionState(option=1, part="A")

        placed = place_questions([lower, _question()], state)

        assert placed.questions[0].placement == QuestionPlacement(1, "B", 1)


class TestJoinedQuestions:
    """A question printed in pieces takes one number, not one per piece."""

    def test_two_joined_pieces_are_written_as_one_question(self) -> None:
        writer = _Writer()
        first = _question()
        first.joins_next = True
        state = PageExtractionState(option=1, part="A")

        placed = place_questions([first, _question()], state, write=writer)

        assert writer.written == [QuestionPlacement(1, "A", 1)]
        assert writer.pieces == [2]
        assert [len(q.regions) for q in placed.questions] == [2]
        assert state.question == 1

    def test_the_pieces_reach_the_writer_in_reading_order(self) -> None:
        writer = _Writer()
        top, bottom = _question(), _question()
        top.joins_next = True

        placed = place_questions(
            [top, bottom], PageExtractionState(option=1, part="A"), write=writer
        )

        assert placed.questions[0].regions == [top, bottom]

    def test_a_piece_at_the_end_of_the_page_is_held_for_the_next_one(self) -> None:
        writer = _Writer()
        whole, piece = _question(), _question()
        piece.joins_next = True
        state = PageExtractionState(option=1, part="A")

        placed = place_questions([whole, piece], state, write=writer)

        assert writer.written == [QuestionPlacement(1, "A", 1)]
        assert placed.held == [piece]
        # The held piece took no number: the page that finishes it numbers it.
        assert state.question == 1

    def test_a_page_that_only_continues_a_question_places_nothing(self) -> None:
        piece = _question()
        piece.joins_next = True
        state = PageExtractionState(option=1, part="A", question=3)

        placed = place_questions([piece], state)

        assert placed.questions == []
        assert placed.held == [piece]
        assert state.question == 3

    def test_a_marker_between_the_pieces_does_not_break_the_join(self) -> None:
        """The pieces of one question are joined however the page is marked up."""
        top, bottom = _question(), _question()
        top.joins_next = True
        state = PageExtractionState(option=1, part="A")

        placed = place_questions([top, _marker("part", "A"), bottom], state)

        assert [len(q.regions) for q in placed.questions] == [2]


class TestCopyRegions:
    def test_a_copy_carries_the_join_a_reviewer_marked(self) -> None:
        region = _question()
        region.joins_next = True
        region.join_offset = (4, -8)

        copy = copy_regions([region])[0]

        assert (copy.joins_next, copy.join_offset) == (True, (4, -8))

    def test_a_copy_cannot_be_reached_through_the_original(self) -> None:
        region = _question()

        copy = copy_regions([region])[0]
        copy.joins_next = True

        assert region.joins_next is False


class TestReadingOrderKey:
    def test_sorts_top_to_bottom_then_left_to_right(self) -> None:
        top_right = PixelPolygon([(500, 10), (600, 10), (600, 50), (500, 50)])
        bottom_left = PixelPolygon([(10, 900), (100, 900), (100, 950), (10, 950)])
        top_left = PixelPolygon([(10, 10), (100, 10), (100, 50), (10, 50)])

        assert sorted([bottom_left, top_right, top_left], key=reading_order_key) == [
            top_left,
            top_right,
            bottom_left,
        ]
