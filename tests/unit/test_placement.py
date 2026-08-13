"""Tests for the question-numbering state machine and the placement walk.

Both are pure, so nothing here touches a page image, a model or the disk: the
walk's only contact with the outside world is the writer it is handed, and
these tests hand it a recorder.
"""

from dataclasses import replace

import pytest

from digitex.domain.entities import PixelPolygon
from digitex.extractors.placement import (
    PageExtractionState,
    PageLabel,
    PageRegion,
    QuestionPlacement,
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

    def __call__(self, region: PageRegion, placement: QuestionPlacement) -> None:
        self.written.append(placement)


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

    def test_placements_number_sequentially_after_commit(self) -> None:
        state = PageExtractionState(option=1, part="A")
        assert state.next_question() == QuestionPlacement(option=1, part="A", number=1)
        state.commit_question()
        assert state.next_question() == QuestionPlacement(option=1, part="A", number=2)

    def test_next_question_without_commit_does_not_consume(self) -> None:
        state = PageExtractionState(option=1, part="A")
        assert state.next_question().number == 1
        assert state.next_question().number == 1

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

        assert [p.placement for p in placed] == [
            QuestionPlacement(1, "A", 1),
            QuestionPlacement(1, "A", 2),
        ]
        assert writer.written == [p.placement for p in placed]
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

        assert [p.placement.number for p in placed] == [5, 6]

    def test_previewing_a_copy_leaves_the_book_state_alone(self) -> None:
        state = PageExtractionState(option=1, part="A", question=4)

        place_questions([_question()], replace(state))

        assert state.question == 4

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

        assert placed[0].placement == QuestionPlacement(1, "B", 1)


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
