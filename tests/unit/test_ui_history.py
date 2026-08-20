"""Tests for the review window's undo timeline.

The window mutates regions in place, so the whole correctness of undo rests on
the history copying what it is given and what it hands back. Both directions
are checked here, because a shared polygon would make undo look like it worked
and then quietly move with the next drag.
"""

from digitex.domain.entities import PixelPolygon
from digitex.domain.placement import (
    PageExtractionState,
    PageLabel,
    PageRegion,
    copy_regions,
)
from digitex.ui.history import EditHistory


def _region(label: PageLabel = "question", x: int = 10) -> PageRegion:
    return PageRegion(
        label=label,
        polygon=PixelPolygon([(x, 10), (x + 20, 10), (x + 20, 30), (x, 30)]),
    )


class TestCopyRegions:
    def test_a_copy_shares_nothing_with_its_original(self) -> None:
        original = [_region()]

        copy = copy_regions(original)
        copy[0].polygon = PixelPolygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        copy[0].label = "part"

        assert original[0].label == "question"
        assert next(iter(original[0].polygon)) == (10, 10)


class TestEditHistory:
    def test_a_fresh_page_has_nothing_to_undo(self) -> None:
        history = EditHistory()
        history.reset([_region()], PageExtractionState())

        assert history.can_undo is False
        assert history.can_redo is False
        assert history.undo() is None

    def test_undo_returns_the_previous_result(self) -> None:
        history = EditHistory()
        regions = [_region()]
        history.reset(regions, PageExtractionState(option=1, part="A"))

        regions.append(_region(x=200))
        history.push(regions, PageExtractionState(option=1, part="A"))

        snapshot = history.undo()

        assert snapshot is not None
        assert len(snapshot.regions) == 1

    def test_redo_walks_back_to_where_undo_started(self) -> None:
        history = EditHistory()
        history.reset([_region()], PageExtractionState())
        history.push([_region(), _region(x=200)], PageExtractionState())

        history.undo()
        snapshot = history.redo()

        assert snapshot is not None
        assert len(snapshot.regions) == 2
        assert history.can_redo is False

    def test_editing_after_an_undo_drops_the_future(self) -> None:
        history = EditHistory()
        history.reset([_region()], PageExtractionState())
        history.push([_region(), _region(x=200)], PageExtractionState())
        history.undo()

        history.push([_region(label="part")], PageExtractionState())

        assert history.can_redo is False

    def test_a_restored_snapshot_is_the_callers_to_edit(self) -> None:
        """Otherwise undoing twice to the same point would return it changed."""
        history = EditHistory()
        history.reset([_region()], PageExtractionState())
        history.push([_region(), _region(x=200)], PageExtractionState())

        first = history.undo()
        assert first is not None
        first.regions[0].label = "option"

        history.redo()
        second = history.undo()

        assert second is not None
        assert second.regions[0].label == "question"

    def test_what_was_pushed_cannot_be_changed_afterwards(self) -> None:
        history = EditHistory()
        regions = [_region()]
        state = PageExtractionState(option=2, part="B", question=4)
        history.reset(regions, state)
        history.push(regions, state)

        regions[0].label = "part"
        state.option = 9

        snapshot = history.undo()

        assert snapshot is not None
        assert snapshot.regions[0].label == "question"
        assert snapshot.state.option == 2

    def test_the_selection_travels_with_the_snapshot(self) -> None:
        history = EditHistory()
        history.reset([_region(), _region(x=200)], PageExtractionState(), selected=1)
        history.push([_region()], PageExtractionState(), selected=None)

        snapshot = history.undo()

        assert snapshot is not None
        assert snapshot.selected == 1

    def test_the_oldest_steps_fall_off_the_end(self) -> None:
        history = EditHistory(limit=3)
        history.reset([_region()], PageExtractionState())
        for count in range(2, 8):
            history.push([_region() for _ in range(count)], PageExtractionState())

        while history.can_undo:
            snapshot = history.undo()

        assert snapshot is not None
        assert len(snapshot.regions) == 5  # the last three results: 5, 6, 7

    def test_pushing_without_a_reset_starts_the_timeline(self) -> None:
        history = EditHistory()

        history.push([_region()], PageExtractionState())

        assert history.can_undo is False
