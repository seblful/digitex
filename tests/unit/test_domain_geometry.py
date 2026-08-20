"""Tests for the conversions between polygon spaces.

Percent is Label Studio's, normalized is what YOLO label files carry, pixels
are what a mask off a page is in. Each space is its own type so a conversion
cannot be applied twice; these check the arithmetic between them.
"""

from __future__ import annotations

from digitex.domain.entities import PercentPolygon, PixelPolygon
from digitex.domain.geometry import percent_to_normalized, pixel_to_percent


class TestCoordinateConversions:
    """Each space is its own type, so a double conversion is a type error.

    A percent polygon fed back into ``percent_to_normalized`` would silently
    divide by 10 000; that it cannot be is checked by ``ty``, not here.
    """

    def test_percent_points_scale_down_to_normalized(self) -> None:
        points = PercentPolygon([[50.0, 100.0], [0.0, 25.0]])

        assert percent_to_normalized(points) == [(0.5, 1.0), (0.0, 0.25)]

    def test_pixel_points_scale_up_against_the_image_size(self) -> None:
        assert pixel_to_percent(PixelPolygon([(320, 240)]), 640, 480) == [[50.0, 50.0]]

    def test_percent_survives_a_round_trip_through_pixels(self) -> None:
        percent = PercentPolygon([[10.0, 20.0], [75.0, 50.0]])

        normalized = percent_to_normalized(percent)
        pixels = PixelPolygon([(int(x * 640), int(y * 480)) for x, y in normalized])

        assert pixel_to_percent(pixels, 640, 480) == percent
