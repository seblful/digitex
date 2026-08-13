"""Tests for the review window's canvas arithmetic.

These are the rules a mouse gesture is interpreted by. A Tk canvas cannot be
asked what it thinks a coordinate means, so the arithmetic lives outside it and
is checked here instead.
"""

import pytest

from digitex.gui.geometry import (
    anchored_origin,
    bounds,
    clamp_scale,
    distance_to_segment,
    fit_scale,
    moved,
    nearest_edge,
    scroll_fraction,
    top_left,
    visible_box,
)

SQUARE = [(10, 10), (30, 10), (30, 30), (10, 30)]


class TestScale:
    def test_fit_uses_whichever_side_runs_out_first(self) -> None:
        # A tall page in a wide canvas is limited by the canvas height.
        assert fit_scale((1000, 2000), (800, 400)) == 0.2

    def test_clamp_holds_the_zoom_between_its_ends(self) -> None:
        assert clamp_scale(50.0, 0.05, 8.0) == 8.0
        assert clamp_scale(0.0001, 0.05, 8.0) == 0.05
        assert clamp_scale(1.5, 0.05, 8.0) == 1.5


class TestVisibleBox:
    def test_a_page_larger_than_the_view_is_clipped_to_it(self) -> None:
        box = visible_box((100.0, 200.0), (800, 600), (4000.0, 5000.0))

        assert box == (100.0, 200.0, 900.0, 800.0)

    def test_a_page_smaller_than_the_view_is_clipped_to_the_page(self) -> None:
        box = visible_box((0.0, 0.0), (800, 600), (300.0, 200.0))

        assert box == (0.0, 0.0, 300.0, 200.0)

    def test_negative_origins_start_at_the_page_edge(self) -> None:
        box = visible_box((-50.0, -20.0), (800, 600), (4000.0, 5000.0))

        assert box == (0.0, 0.0, 750.0, 580.0)

    def test_a_view_scrolled_off_the_page_shows_nothing(self) -> None:
        assert visible_box((900.0, 0.0), (800, 600), (400.0, 400.0)) is None


class TestAnchoredZoom:
    def test_the_point_under_the_cursor_does_not_move(self) -> None:
        """Zooming in on a detail should magnify it, not scroll away from it."""
        origin, pointer, ratio = 100.0, 300.0, 2.0

        new_origin = anchored_origin(origin, pointer, ratio)

        # The cursor sat 200px into the view; it still does, and the image
        # pixel it pointed at (canvas 300 -> 600) is right there.
        assert pointer * ratio - new_origin == pointer - origin
        assert new_origin == 400.0

    def test_zooming_out_at_the_edge_can_ask_for_a_negative_origin(self) -> None:
        """Which the fraction clamps — there is nothing left of the page."""
        origin = anchored_origin(0.0, 50.0, 0.5)

        assert origin < 0
        assert scroll_fraction(origin, 1000) == 0.0

    def test_a_page_with_no_extent_scrolls_nowhere(self) -> None:
        assert scroll_fraction(10.0, 0) == 0.0


class TestPolygons:
    def test_a_point_inserts_into_the_edge_it_was_clicked_on(self) -> None:
        # Just outside the middle of the bottom edge, which runs 2 -> 3.
        assert nearest_edge(SQUARE, (20, 33)) == 2

    def test_the_closing_edge_is_an_edge_like_any_other(self) -> None:
        assert nearest_edge(SQUARE, (7, 20)) == 3

    def test_an_empty_polygon_has_no_edges(self) -> None:
        with pytest.raises(ValueError, match="no edges"):
            nearest_edge([], (0, 0))

    def test_distance_to_a_degenerate_segment_is_to_its_point(self) -> None:
        assert distance_to_segment((3, 4), (0, 0), (0, 0)) == 25

    def test_moving_shifts_every_point(self) -> None:
        assert moved(SQUARE, -10, 5) == [(0, 15), (20, 15), (20, 35), (0, 35)]

    def test_bounds_box_the_whole_polygon(self) -> None:
        assert bounds([(5, 40), (30, 10), (12, 60)]) == (5, 10, 30, 60)

    def test_a_caption_hangs_off_the_topmost_point(self) -> None:
        assert top_left([(30, 10), (5, 40), (12, 10)]) == (12, 10)

    def test_an_empty_polygon_has_no_bounds(self) -> None:
        with pytest.raises(ValueError, match="no bounds"):
            bounds([])
