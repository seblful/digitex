"""Tests for the Label Studio geometry seam."""

from digitex.domain.entities import PercentPolygon, PixelPolygon
from digitex.labeling.geometry import (
    local_file_path,
    percent_to_normalized,
    pixel_to_percent,
)


class TestLocalFilePath:
    """Test local-files URI parsing."""

    def test_d_parameter(self) -> None:
        uri = "/data/local-files/?d=training%5Cdata%5Cimages%5Cbiology_2008_12.jpg"
        path = local_file_path(uri)
        assert path is not None
        assert path.name == "biology_2008_12.jpg"

    def test_file_parameter(self) -> None:
        uri = "/data/local-files/?file=training/data/images/page.jpg"
        path = local_file_path(uri)
        assert path is not None
        assert path.name == "page.jpg"

    def test_url_encoded_spaces(self) -> None:
        uri = "/data/local-files/?d=images%5Cmy%20file.jpg"
        path = local_file_path(uri)
        assert path is not None
        assert path.name == "my file.jpg"

    def test_backslash_uri_splits_on_every_platform(self) -> None:
        """The separators are the Label Studio host's, not this machine's.

        Asserting ``.name`` alone passed on Windows while the whole URI stayed
        one filename on Linux, which is how this reached CI unnoticed.
        """
        uri = "/data/local-files/?d=training%5Cdata%5Cimages%5Cpage.jpg"
        path = local_file_path(uri)
        assert path is not None
        assert path.parts == ("training", "data", "images", "page.jpg")

    def test_empty_uri(self) -> None:
        assert local_file_path("") is None

    def test_no_local_file_parameter(self) -> None:
        assert local_file_path("http://example.com/image.jpg") is None


class TestCoordinateConversions:
    """Test percent/normalized/pixel polygon conversions.

    Each space is its own type, so a percent polygon fed back into
    ``percent_to_normalized`` is a type error rather than a silent divide by
    10 000. That guarantee is checked by ``ty``, not by a test here.
    """

    def test_percent_to_normalized(self) -> None:
        points = PercentPolygon([[50.0, 100.0], [0.0, 25.0]])
        assert percent_to_normalized(points) == [(0.5, 1.0), (0.0, 0.25)]

    def test_pixel_to_percent(self) -> None:
        pixels = PixelPolygon([(320, 240)])
        assert pixel_to_percent(pixels, 640, 480) == [[50.0, 50.0]]

    def test_round_trip_percent_to_pixel_and_back(self) -> None:
        percent = PercentPolygon([[10.0, 20.0], [75.0, 50.0]])
        normalized = percent_to_normalized(percent)
        pixels = PixelPolygon([(int(x * 640), int(y * 480)) for x, y in normalized])
        assert pixel_to_percent(pixels, 640, 480) == percent
