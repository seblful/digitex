"""Tests for edge cases in image processing."""

import numpy as np
import pytest
from PIL import Image

from digitex.core.processors.image import ImageCropper, SegmentProcessor


class TestImageCropperEdgeCases:
    """Test suite for ImageCropper edge cases."""

    def test_polygon_to_quad_square(self) -> None:
        """Test _polygon_to_quad with perfect square."""
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        quad = ImageCropper._polygon_to_quad(polygon)
        assert quad.shape == (4, 2)

    def test_polygon_to_quad_rotated(self) -> None:
        """Test _polygon_to_quad with rotated rectangle."""
        polygon = [(10, 5), (60, 15), (50, 70), (0, 60)]
        quad = ImageCropper._polygon_to_quad(polygon)
        assert quad.shape == (4, 2)

    def test_order_quad_points_all_same(self) -> None:
        """Test _order_quad_points with all points the same (degenerate)."""
        pts = np.array([[50, 50], [50, 50], [50, 50], [50, 50]], dtype=np.float32)
        ordered = ImageCropper._order_quad_points(pts)
        assert ordered.shape == (4, 2)

    def test_order_quad_points_line(self) -> None:
        """Test _order_quad_points with collinear points."""
        pts = np.array([[0, 0], [50, 0], [100, 0], [150, 0]], dtype=np.float32)
        ordered = ImageCropper._order_quad_points(pts)
        assert ordered.shape == (4, 2)

    def test_perspective_transform_dimensions(self) -> None:
        """Test _perspective_transform calculates correct output dimensions."""
        pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        w, h, _ = ImageCropper._perspective_transform(pts)
        assert w == 100
        assert h == 50

    def test_perspective_transform_trapezoid(self) -> None:
        """Test _perspective_transform with trapezoid input."""
        pts = np.array([[10, 0], [90, 0], [100, 50], [0, 50]], dtype=np.float32)
        w, h, _ = ImageCropper._perspective_transform(pts)
        assert w >= 90
        assert h == 50

    def test_cut_out_image_by_polygon_with_many_points(self) -> None:
        """Test cut_out_image_by_polygon with polygon having many points."""
        img = Image.new("RGB", (200, 200), color="white")
        polygon = [
            (20, 20),
            (50, 20),
            (80, 20),
            (80, 50),
            (80, 80),
            (50, 80),
            (20, 80),
            (20, 50),
        ]
        result = ImageCropper.cut_out_image_by_polygon(img, polygon)
        assert result.mode == "RGBA"
        assert result.size[0] > 0
        assert result.size[1] > 0

    def test_cut_out_image_by_polygon_complex_shape(self) -> None:
        """Test cut_out_image_by_polygon with complex polygon."""
        img = Image.new("RGB", (300, 300), color="white")
        polygon = [(25, 25), (75, 20), (80, 50), (70, 80), (30, 75), (20, 50)]
        result = ImageCropper.cut_out_image_by_polygon(img, polygon)
        assert result.mode == "RGBA"
        assert result.size[0] > 0
        assert result.size[1] > 0


class TestSegmentProcessorEdgeCases:
    """Test suite for SegmentProcessor edge cases."""

    def test_remove_color_all_saturated(self) -> None:
        """Test remove_color when all pixels are saturated (pure red)."""
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        img[:, :, 0] = 255
        img[:, :, 3] = 255
        result = SegmentProcessor.remove_color(img, saturation_threshold=100)
        assert np.all(result[:, :, 3] == 0)

    def test_remove_color_all_grayscale(self) -> None:
        """Test remove_color when all pixels are grayscale."""
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        img[:, :, :3] = 128
        img[:, :, 3] = 255
        result = SegmentProcessor.remove_color(img, saturation_threshold=100)
        assert np.all(result[:, :, 3] == 255)

    def test_remove_color_high_threshold(self) -> None:
        """Test remove_color with very high saturation threshold (only highly saturated colors removed)."""
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        img[:, :, 0] = 200
        img[:, :, 1] = 200
        img[:, :, 2] = 200
        img[:, :, 3] = 255
        result = SegmentProcessor.remove_color(img, saturation_threshold=200)
        assert np.all(result[:, :, 3] == 255)

    def test_remove_bg_extremely_bright(self) -> None:
        """Test remove_bg with extremely bright image (value > threshold -> removed)."""
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        img[:, :, :3] = 254
        img[:, :, 3] = 255
        result = SegmentProcessor.remove_bg(img, threshold=200)
        assert np.all(result[:, :, 3] == 0)

    def test_remove_bg_extremely_dark(self) -> None:
        """Test remove_bg with extremely dark image (value <= threshold -> kept)."""
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        img[:, :, 3] = 255
        result = SegmentProcessor.remove_bg(img, threshold=50)
        assert np.all(result[:, :, 3] == 255)

    def test_remove_bg_boundary_threshold(self) -> None:
        """Test remove_bg with threshold at boundary (0 and 255)."""
        img = np.full((50, 50, 4), 128, dtype=np.uint8)
        img[:, :, 3] = 255

        result_0 = SegmentProcessor.remove_bg(img, threshold=0)
        assert np.all(result_0[:, :, 3] == 0)

        result_255 = SegmentProcessor.remove_bg(img, threshold=255)
        assert np.all(result_255[:, :, 3] == 255)

    def test_increase_darkness_very_small_gamma(self) -> None:
        """Test increase_darkness with very small gamma (very dark)."""
        img = np.full((10, 10, 4), 200, dtype=np.uint8)
        img[:, :, 3] = 255
        result = SegmentProcessor.increase_darkness(img, gamma=0.1)
        assert result[0, 0, 0] < 200
        assert result[0, 0, 3] == 255

    def test_increase_darkness_very_large_gamma(self) -> None:
        """Test increase_darkness with very large gamma (very bright)."""
        img = np.full((10, 10, 4), 50, dtype=np.uint8)
        img[:, :, 3] = 255
        result = SegmentProcessor.increase_darkness(img, gamma=10.0)
        assert result[0, 0, 0] > 50
        assert result[0, 0, 3] == 255

    def test_add_white_background_all_transparent(self) -> None:
        """Test add_white_background when image is fully transparent."""
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        result = SegmentProcessor.add_white_background(img)
        assert result.shape == (10, 10, 3)
        assert np.all(result == 255)

    def test_add_white_background_all_opaque(self) -> None:
        """Test add_white_background when image is fully opaque."""
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[:, :, :3] = [100, 150, 200]
        img[:, :, 3] = 255
        result = SegmentProcessor.add_white_background(img)
        assert result.shape == (10, 10, 3)
        np.testing.assert_array_equal(result[:, :, 0], 100)
        np.testing.assert_array_equal(result[:, :, 1], 150)
        np.testing.assert_array_equal(result[:, :, 2], 200)

    def test_add_white_background_partial_transparency(self) -> None:
        """Test add_white_background with partial transparency."""
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[:, :, :3] = [255, 0, 0]
        img[:, :, 3] = 128
        result = SegmentProcessor.add_white_background(img)
        assert result.shape == (10, 10, 3)
        assert result[0, 0, 0] > 127
        assert result[0, 0, 1] < 128
        assert result[0, 0, 2] < 128

    def test_process_with_custom_parameters(self) -> None:
        """Test process with custom saturation, bg_threshold, and gamma."""
        img = Image.new("RGBA", (50, 50), color=(128, 128, 128, 255))
        processor = SegmentProcessor()
        result = processor.process(
            img,
            saturation_threshold=50,
            bg_threshold=100,
            gamma=0.5,
        )
        assert result.mode == "RGB"
        assert result.size == (50, 50)

    def test_process_with_zero_threshold(self) -> None:
        """Test process with zero threshold values."""
        img = Image.new("RGBA", (50, 50), color=(0, 0, 0, 255))
        processor = SegmentProcessor()
        result = processor.process(
            img,
            saturation_threshold=0,
            bg_threshold=0,
            gamma=0.5,
        )
        assert result.mode == "RGB"
        assert result.size == (50, 50)
