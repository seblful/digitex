"""Tests for snapping a hand-traced outline onto the print it contains.

The subject is a synthetic page: solid black bars on white, which stand in for
lines of print. Synthetic because the properties being asserted are the ones the
module promises for *any* page — that print is never dropped, that two outlines
never overlap, that the shape is not thrown away — and a real scan would make
each of those a claim about one book rather than about the algorithm.

Every test states what a loose or a tight outline should become, never a
coordinate: the tuning constants are expected to move, and a test that pins
0.25 line heights to a pixel would have to move with them.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from digitex.domain.entities import PixelPolygon
from digitex.imaging import outlines
from digitex.imaging.outlines import Outline, align_outlines, tangled

# A bar this tall reads as a line of print rather than as a speck or a fragment.
BAR_HEIGHT = 30
PAGE = (900, 600)  # width, height


def _page(bars: list[tuple[int, int, int, int]]) -> Image.Image:
    """A white page with a solid black bar for each (left, top, right, bottom)."""
    pixels = np.full((PAGE[1], PAGE[0]), 255, dtype=np.uint8)
    for left, top, right, bottom in bars:
        pixels[top:bottom, left:right] = 0
    return Image.fromarray(pixels, mode="L")


def _box(left: int, top: int, right: int, bottom: int) -> PixelPolygon:
    return PixelPolygon([(left, top), (right, top), (right, bottom), (left, bottom)])


def _bounds(polygon: PixelPolygon) -> tuple[int, int, int, int]:
    points = np.array(polygon)
    return (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()),
        int(points[:, 1].max()),
    )


def _ink_inside(image: Image.Image, polygon: PixelPolygon) -> int:
    """How many of the page's black pixels the polygon encloses."""
    import cv2

    mask = np.zeros((PAGE[1], PAGE[0]), np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)
    return int(((np.array(image) < 128) & (mask > 0)).sum())


class TestTightening:
    def test_a_loose_outline_is_pulled_in_to_the_print(self) -> None:
        """The complaint the module exists for: slack the mouse left behind."""
        bar = (100, 100, 500, 100 + BAR_HEIGHT)
        page = _page([bar])
        loose = Outline("question", _box(20, 20, 800, 300))

        aligned = align_outlines(page, [loose])

        assert aligned[0].changed, aligned[0].reason
        # Pulled in on every side, and still outside the print on every side.
        left, top, right, bottom = _bounds(aligned[0].polygon)
        assert (left, top) > (20, 20)
        assert (right, bottom) < (800, 300)
        assert (left, top) < (bar[0], bar[1])
        assert (right, bottom) > (bar[2], bar[3])

    def test_a_tight_outline_is_pushed_out_to_the_margin(self) -> None:
        """Normalising cuts both ways: 1667 of the corpus's regions grew."""
        bar = (100, 100, 500, 100 + BAR_HEIGHT)
        page = _page([bar])
        # Drawn exactly on the print, leaving no margin at all.
        clipping = Outline("question", _box(*bar))

        aligned = align_outlines(page, [clipping])

        assert aligned[0].changed, aligned[0].reason
        left, top, right, bottom = _bounds(aligned[0].polygon)
        assert (left, top) < (bar[0], bar[1])
        assert (right, bottom) > (bar[2], bar[3])

    def test_two_regions_end_up_the_same_distance_from_their_print(self) -> None:
        """One loose, one tight, over identical print: the same margin out."""
        page = _page(
            [
                (100, 100, 500, 100 + BAR_HEIGHT),
                (100, 300, 500, 300 + BAR_HEIGHT),
            ]
        )
        aligned = align_outlines(
            page,
            [
                Outline("question", _box(30, 40, 700, 180)),
                Outline("question", _box(100, 300, 500, 300 + BAR_HEIGHT)),
            ],
        )

        assert all(item.changed for item in aligned), [i.reason for i in aligned]
        gaps = []
        for item, bar_top in zip(aligned, (100, 300), strict=True):
            left, top, right, _ = _bounds(item.polygon)
            gaps.append((100 - left, bar_top - top, right - 500))
        assert gaps[0] == gaps[1]


class TestPrintIsNeverDropped:
    def test_every_pixel_of_print_survives_the_rebuild(self) -> None:
        page = _page(
            [
                (80, 60, 520, 60 + BAR_HEIGHT),
                (80, 110, 300, 110 + BAR_HEIGHT),
                (80, 160, 460, 160 + BAR_HEIGHT),
            ]
        )
        loose = Outline("question", _box(20, 20, 800, 260))

        aligned = align_outlines(page, [loose])

        assert aligned[0].changed, aligned[0].reason
        assert _ink_inside(page, aligned[0].polygon) == _ink_inside(page, loose.polygon)

    def test_a_short_line_keeps_its_own_step(self) -> None:
        """A line half the width of its neighbours should not be squared off."""
        page = _page(
            [
                (80, 60, 520, 60 + BAR_HEIGHT),
                (80, 110, 200, 110 + BAR_HEIGHT),
                (80, 160, 520, 160 + BAR_HEIGHT),
            ]
        )
        aligned = align_outlines(page, [Outline("question", _box(20, 20, 800, 260))])

        assert aligned[0].changed, aligned[0].reason
        # A plain rectangle would be four points; a step costs more than that.
        assert len(aligned[0].polygon) > 4


class TestRegionsStayApart:
    def test_neighbouring_regions_do_not_overlap(self) -> None:
        """Two loose outlines over two lines, each reaching past the other."""
        import cv2

        page = _page(
            [
                (100, 100, 500, 100 + BAR_HEIGHT),
                (100, 200, 500, 200 + BAR_HEIGHT),
            ]
        )
        aligned = align_outlines(
            page,
            [
                Outline("question", _box(40, 40, 700, 190)),
                Outline("question", _box(40, 150, 700, 320)),
            ],
        )

        masks = []
        for item in aligned:
            mask = np.zeros((PAGE[1], PAGE[0]), np.uint8)
            cv2.fillPoly(mask, [np.array(item.polygon, np.int32)], 255)
            masks.append(mask > 0)
        assert not (masks[0] & masks[1]).any()

    def test_a_region_gives_up_print_its_neighbour_mostly_holds(self) -> None:
        """A line reached into but not really held goes to the region holding it.

        The first outline sags over the top eight rows of the second line; the
        second outline holds all thirty. The line is the second region's, so the
        first must come back off it.
        """
        page = _page(
            [
                (100, 100, 500, 100 + BAR_HEIGHT),
                (100, 200, 500, 200 + BAR_HEIGHT),
            ]
        )
        overreaching = _box(40, 40, 700, 208)
        aligned = align_outlines(
            page,
            [
                Outline("question", overreaching),
                Outline("question", _box(90, 190, 520, 260)),
            ],
        )

        assert all(item.changed for item in aligned), [i.reason for i in aligned]
        assert _ink_inside(page, aligned[0].polygon) < _ink_inside(page, overreaching)
        # And the region that owns the line keeps every pixel of it.
        assert _ink_inside(page, aligned[1].polygon) == BAR_HEIGHT * 400

    def test_two_outlines_over_the_same_print_do_not_both_claim_it(self) -> None:
        """A duplicate region is an annotator's slip, and is reported as one.

        Print goes to whichever outline holds most of it, and when two hold all
        of it the tie goes to the first. The second is then a region with no print
        of its own — which is exactly what it is, and what it gets told.
        """
        page = _page([(100, 100, 500, 100 + BAR_HEIGHT)])
        aligned = align_outlines(
            page,
            [
                Outline("question", _box(60, 60, 600, 200)),
                Outline("question", _box(70, 70, 590, 190)),
            ],
        )

        assert aligned[0].changed
        assert not aligned[1].changed
        assert aligned[1].reason == "no print of its own"


class TestRefusals:
    def test_a_region_holding_no_print_is_left_exactly_as_found(self) -> None:
        page = _page([(100, 100, 500, 100 + BAR_HEIGHT)])
        blank = Outline("question", _box(600, 400, 800, 500))

        aligned = align_outlines(page, [blank])

        assert not aligned[0].changed
        assert aligned[0].reason == "no print of its own"
        assert aligned[0].polygon == blank.polygon

    def test_no_outlines_is_not_a_page_to_read(self) -> None:
        """Cheap enough to be worth not paying for: no ink pass, no skew search."""
        assert align_outlines(_page([]), []) == []

    def test_the_label_always_travels_unchanged(self) -> None:
        page = _page([(100, 100, 500, 100 + BAR_HEIGHT)])
        aligned = align_outlines(
            page,
            [
                Outline("question", _box(20, 20, 700, 200)),
                Outline("part", _box(600, 400, 800, 500)),
            ],
        )
        assert [item.label for item in aligned] == ["question", "part"]


class TestRingsAreUsable:
    def test_every_rebuilt_ring_is_a_simple_polygon_on_the_page(self) -> None:
        page = _page(
            [
                (80, 60, 520, 60 + BAR_HEIGHT),
                (80, 110, 200, 110 + BAR_HEIGHT),
                (300, 160, 560, 160 + BAR_HEIGHT),
                (80, 210, 460, 210 + BAR_HEIGHT),
            ]
        )
        aligned = align_outlines(page, [Outline("question", _box(20, 20, 850, 300))])

        polygon = aligned[0].polygon
        assert len(polygon) >= outlines.MIN_RING_POINTS
        assert not tangled(np.array(polygon, dtype=np.float64))
        assert all(0 <= x <= PAGE[0] and 0 <= y <= PAGE[1] for x, y in polygon)

    @pytest.mark.parametrize(
        ("ring", "expected"),
        [
            ([(0, 0), (10, 0), (10, 10), (0, 10)], False),
            ([(0, 0), (10, 10), (10, 0), (0, 10)], True),
            ([(0, 0), (10, 0), (10, 10)], False),
        ],
        ids=["square", "bowtie", "triangle"],
    )
    def test_tangled_knows_a_bowtie_from_a_square(
        self, ring: list[tuple[int, int]], expected: bool
    ) -> None:
        assert tangled(np.array(ring, dtype=np.float64)) is expected

    def test_a_ring_touching_itself_at_a_point_is_not_a_crossing(self) -> None:
        """A vertex landing on a non-adjacent edge still fills correctly.

        Treating it as a crossing is what made an earlier version reject good
        outlines: once a ring is rotated, the exact zeros meaning "collinear"
        become noise a hair either side of it.
        """
        ring = np.array([[33.0, 3.0], [12.0, 17.0], [28.0, 33.0], [3.0, 8.0]])
        assert tangled(ring) is False
