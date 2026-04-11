"""Tests for the YOLO converter module."""

import numpy as np

from digitex.ml.yolo.converter import Converter


class TestConverter:
    """Test suite for Converter class."""

    def test_point_to_polygon(self) -> None:
        """Test point_to_polygon converts normalized coordinates to pixel coordinates."""
        point = [0.1, 0.2, 0.5, 0.5, 0.5, 0.8, 0.1, 0.8]
        img_width = 100
        img_height = 200

        result = Converter.point_to_polygon(point, img_width, img_height)

        expected = np.array(
            [
                [10.0, 40.0],
                [50.0, 100.0],
                [50.0, 160.0],
                [10.0, 160.0],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_point_to_polygon_single_point(self) -> None:
        """Test point_to_polygon with single point coordinates."""
        point = [0.5, 0.5]
        img_width = 100
        img_height = 100

        result = Converter.point_to_polygon(point, img_width, img_height)

        expected = np.array([[50.0, 50.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_point_to_polygon_full_image(self) -> None:
        """Test point_to_polygon with coordinates spanning full image."""
        point = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        img_width = 1000
        img_height = 500

        result = Converter.point_to_polygon(point, img_width, img_height)

        expected = np.array(
            [
                [0.0, 0.0],
                [1000.0, 0.0],
                [1000.0, 500.0],
                [0.0, 500.0],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_polygon_to_point(self) -> None:
        """Test polygon_to_point converts pixel coordinates to normalized coordinates."""
        polygon = np.array(
            [
                [10.0, 40.0],
                [50.0, 100.0],
                [50.0, 160.0],
                [10.0, 160.0],
            ]
        )
        img_width = 100
        img_height = 200

        result = Converter.polygon_to_point(polygon, img_width, img_height)

        expected = [0.1, 0.2, 0.5, 0.5, 0.5, 0.8, 0.1, 0.8]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-6

    def test_polygon_to_point_single_point(self) -> None:
        """Test polygon_to_point with single point."""
        polygon = np.array([[50.0, 50.0]])
        img_width = 100
        img_height = 100

        result = Converter.polygon_to_point(polygon, img_width, img_height)

        expected = [0.5, 0.5]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-6

    def test_polygon_to_point_full_image(self) -> None:
        """Test polygon_to_point with full image coordinates."""
        polygon = np.array(
            [
                [0.0, 0.0],
                [1000.0, 0.0],
                [1000.0, 500.0],
                [0.0, 500.0],
            ]
        )
        img_width = 1000
        img_height = 500

        result = Converter.polygon_to_point(polygon, img_width, img_height)

        expected = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-6

    def test_roundtrip_conversion(self) -> None:
        """Test that point_to_polygon and polygon_to_point are inverses."""
        original = [0.123, 0.456, 0.789, 0.123, 0.789, 0.654, 0.123, 0.654]
        img_width = 1920
        img_height = 1080

        polygon = Converter.point_to_polygon(original, img_width, img_height)
        result = Converter.polygon_to_point(polygon, img_width, img_height)

        for r, o in zip(result, original):
            assert abs(r - o) < 1e-6
