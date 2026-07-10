"""Tests for the Label Studio geometry seam."""

from digitex.label_studio.geometry import (
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

    def test_empty_uri(self) -> None:
        assert local_file_path("") is None

    def test_no_local_file_parameter(self) -> None:
        assert local_file_path("http://example.com/image.jpg") is None


class TestCoordinateConversions:
    """Test percent/normalized/pixel polygon conversions."""

    def test_percent_to_normalized(self) -> None:
        points = [[50.0, 100.0], [0.0, 25.0]]
        assert percent_to_normalized(points) == [(0.5, 1.0), (0.0, 0.25)]

    def test_pixel_to_percent(self) -> None:
        assert pixel_to_percent([(320, 240)], 640, 480) == [[50.0, 50.0]]

    def test_round_trip_percent_to_pixel_and_back(self) -> None:
        percent = [[10.0, 20.0], [75.0, 50.0]]
        normalized = percent_to_normalized(percent)
        pixels = [(int(x * 640), int(y * 480)) for x, y in normalized]
        assert pixel_to_percent(pixels, 640, 480) == percent
