"""Tests for the Label Studio geometry seam.

Two things cross this seam: the URIs Label Studio references local images by,
and the percent space it stores polygon points in. Both are produced by a
server that may not be running the same OS as the machine reading them, which
is what the separator handling below is about.
"""

from __future__ import annotations

import pytest

from digitex.domain.entities import PercentPolygon, PixelPolygon
from digitex.domain.geometry import (
    local_file_path,
    percent_to_normalized,
    pixel_to_percent,
)


class TestLocalFilePath:
    @pytest.mark.parametrize(
        ("uri", "name"),
        [
            ("/data/local-files/?d=training%5Cdata%5Cpage.jpg", "page.jpg"),
            ("/data/local-files/?file=training/data/page.jpg", "page.jpg"),
            ("/data/local-files/?d=images%5Cmy%20file.jpg", "my file.jpg"),
        ],
        ids=["d-parameter", "file-parameter", "url-encoded-space"],
    )
    def test_the_filename_is_recovered_from_either_parameter(
        self, uri: str, name: str
    ) -> None:
        path = local_file_path(uri)

        assert path is not None
        assert path.name == name

    def test_a_backslash_uri_splits_on_every_platform(self) -> None:
        """The separators are the Label Studio host's, not this machine's.

        Asserting ``.name`` alone passed on Windows while the whole URI stayed
        one filename on Linux, which is how this reached CI unnoticed.
        """
        uri = "/data/local-files/?d=training%5Cdata%5Cimages%5Cpage.jpg"

        path = local_file_path(uri)

        assert path is not None
        assert path.parts == ("training", "data", "images", "page.jpg")

    @pytest.mark.parametrize(
        "uri",
        ["", "http://example.com/image.jpg", "/data/local-files/?other=x"],
        ids=["empty", "remote-url", "no-local-file-parameter"],
    )
    def test_a_uri_naming_no_local_file_has_no_path(self, uri: str) -> None:
        """The predictor skips such a task rather than failing the run."""
        assert local_file_path(uri) is None


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
