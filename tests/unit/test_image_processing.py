"""Tests for the image processors: SegmentProcessor, ImageCropper, resize_image."""

import numpy as np
import pytest
from PIL import Image

from digitex.core.processors import SegmentProcessor, resize_image
from digitex.core.processors.image import (
    ImageCropper,
    _order_quad_points,
    _perspective_transform,
    _polygon_to_quad,
)


class TestSegmentProcessorRemoveColor:
    def test_removes_saturated_pixels(self) -> None:
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[:, :, :3] = [0, 0, 0]
        img[:, :, 3] = 255
        img[10:20, 10:20] = [255, 0, 0, 255]

        result = SegmentProcessor.remove_color(img, saturation_threshold=100)
        assert result[15, 15, 3] == 0
        assert result[5, 5, 3] == 255

    def test_preserves_grayscale(self) -> None:
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[:, :, :3] = [128, 128, 128]
        img[:, :, 3] = 255

        result = SegmentProcessor.remove_color(img, saturation_threshold=100)
        assert result[50, 50, 3] == 255

    def test_no_dilation(self) -> None:
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[:, :, 3] = 255
        img[10:20, 10:20] = [255, 0, 0, 255]

        result = SegmentProcessor.remove_color(
            img, saturation_threshold=100, dilate_iterations=0
        )
        assert result[15, 15, 3] == 0

    def test_all_saturated(self) -> None:
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        img[:, :, 0] = 255
        img[:, :, 3] = 255
        result = SegmentProcessor.remove_color(img, saturation_threshold=100)
        assert np.all(result[:, :, 3] == 0)

    def test_all_grayscale(self) -> None:
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        img[:, :, :3] = 128
        img[:, :, 3] = 255
        result = SegmentProcessor.remove_color(img, saturation_threshold=100)
        assert np.all(result[:, :, 3] == 255)

    def test_high_threshold_keeps_low_saturation(self) -> None:
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        img[:, :, 0] = 200
        img[:, :, 1] = 200
        img[:, :, 2] = 200
        img[:, :, 3] = 255
        result = SegmentProcessor.remove_color(img, saturation_threshold=200)
        assert np.all(result[:, :, 3] == 255)


class TestSegmentProcessorRemoveBg:
    def test_returns_rgba_with_transparent_bright_pixels(self) -> None:
        processor = SegmentProcessor()
        img = np.ones((100, 100, 4), dtype=np.uint8) * 255
        img[20:80, 20:80, :3] = [50, 50, 50]
        result = processor.remove_bg(img)

        assert result.shape == (100, 100, 4)
        assert result.dtype == np.uint8
        alpha = result[:, :, 3]
        assert alpha[0, 0] == 0
        assert alpha[50, 50] == 255

    @pytest.mark.parametrize("threshold", [300, -1], ids=["above-255", "negative"])
    def test_invalid_threshold_raises(self, threshold: int) -> None:
        processor = SegmentProcessor()
        img = np.ones((100, 100, 4), dtype=np.uint8) * 255
        with pytest.raises(
            ValueError, match="Threshold must be an integer in range 0-255"
        ):
            processor.remove_bg(img, threshold=threshold)

    def test_extremely_bright_image_fully_removed(self) -> None:
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        img[:, :, :3] = 254
        img[:, :, 3] = 255
        result = SegmentProcessor.remove_bg(img, threshold=200)
        assert np.all(result[:, :, 3] == 0)

    def test_extremely_dark_image_fully_kept(self) -> None:
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        img[:, :, 3] = 255
        result = SegmentProcessor.remove_bg(img, threshold=50)
        assert np.all(result[:, :, 3] == 255)

    def test_boundary_thresholds(self) -> None:
        img = np.full((50, 50, 4), 128, dtype=np.uint8)
        img[:, :, 3] = 255

        result_0 = SegmentProcessor.remove_bg(img, threshold=0)
        assert np.all(result_0[:, :, 3] == 0)

        result_255 = SegmentProcessor.remove_bg(img, threshold=255)
        assert np.all(result_255[:, :, 3] == 255)


class TestSegmentProcessorIncreaseDarkness:
    def test_darkens_midtones(self) -> None:
        processor = SegmentProcessor()
        img = np.full((10, 10, 4), 128, dtype=np.uint8)
        img[:, :, 3] = 255

        result = processor.increase_darkness(img, gamma=0.8)

        assert result[0, 0, 0] < 128
        assert result[0, 0, 3] == 255

    def test_gamma_one_leaves_image_unchanged(self) -> None:
        processor = SegmentProcessor()
        img = np.full((10, 10, 4), 100, dtype=np.uint8)
        img[:, :, 3] = 255

        result = processor.increase_darkness(img, gamma=1.0)

        np.testing.assert_array_equal(result, img)

    def test_preserves_alpha(self) -> None:
        processor = SegmentProcessor()
        img = np.full((10, 10, 4), 100, dtype=np.uint8)
        img[:, :, 3] = 128

        result = processor.increase_darkness(img, gamma=0.8)

        assert result[0, 0, 3] == 128

    @pytest.mark.parametrize("gamma", [0, -1], ids=["zero", "negative"])
    def test_invalid_gamma_raises(self, gamma: float) -> None:
        processor = SegmentProcessor()
        img = np.full((10, 10, 4), 100, dtype=np.uint8)
        img[:, :, 3] = 255

        with pytest.raises(ValueError, match="gamma must be positive"):
            processor.increase_darkness(img, gamma=gamma)

    def test_very_small_gamma_darkens(self) -> None:
        img = np.full((10, 10, 4), 200, dtype=np.uint8)
        img[:, :, 3] = 255
        result = SegmentProcessor.increase_darkness(img, gamma=0.1)
        assert result[0, 0, 0] < 200
        assert result[0, 0, 3] == 255

    def test_very_large_gamma_brightens(self) -> None:
        img = np.full((10, 10, 4), 50, dtype=np.uint8)
        img[:, :, 3] = 255
        result = SegmentProcessor.increase_darkness(img, gamma=10.0)
        assert result[0, 0, 0] > 50
        assert result[0, 0, 3] == 255


class TestSegmentProcessorAddWhiteBackground:
    def test_transparent_becomes_white(self) -> None:
        processor = SegmentProcessor()
        img = np.full((10, 10, 4), 100, dtype=np.uint8)
        img[:, :, 3] = 0

        result = processor.add_white_background(img)

        assert result.shape[2] == 3
        assert result[0, 0, 0] == 255
        assert result[0, 0, 1] == 255
        assert result[0, 0, 2] == 255

    def test_opaque_unchanged(self) -> None:
        processor = SegmentProcessor()
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[:, :, 0] = 50
        img[:, :, 1] = 100
        img[:, :, 2] = 150
        img[:, :, 3] = 255

        result = processor.add_white_background(img)

        assert result[0, 0, 0] == 50
        assert result[0, 0, 1] == 100
        assert result[0, 0, 2] == 150

    def test_fully_transparent_image_is_all_white(self) -> None:
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        result = SegmentProcessor.add_white_background(img)
        assert result.shape == (10, 10, 3)
        assert np.all(result == 255)

    def test_partial_transparency_blends_toward_white(self) -> None:
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[:, :, :3] = [255, 0, 0]
        img[:, :, 3] = 128
        result = SegmentProcessor.add_white_background(img)
        assert result.shape == (10, 10, 3)
        assert result[0, 0, 0] > 127
        assert result[0, 0, 1] < 128
        assert result[0, 0, 2] < 128


class TestSegmentProcessorProcess:
    def test_process_with_custom_parameters(self) -> None:
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

    def test_process_with_zero_thresholds(self) -> None:
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


class TestResizeImage:
    def test_pads_smaller_image_to_max_dimensions(self) -> None:
        img = Image.new("RGB", (100, 100), color="red")
        result = resize_image(img, 200, 200)
        assert result.size == (200, 200)

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ((400, 200), (200, 100)),
            ((200, 400), (100, 200)),
        ],
        ids=["width-limited", "height-limited"],
    )
    def test_preserves_aspect_ratio(
        self, source: tuple[int, int], expected: tuple[int, int]
    ) -> None:
        img = Image.new("RGB", source, color="red")
        result = resize_image(img, 200, 200)
        assert result.size == expected

    def test_landscape_shrinks_to_fit(self) -> None:
        img = Image.new("RGB", (200, 100), color="red")
        result = resize_image(img, 100, 100)
        assert result.size == (100, 50)

    def test_portrait_shrinks_to_fit(self) -> None:
        img = Image.new("RGB", (100, 200), color="red")
        result = resize_image(img, 100, 100)
        assert result.size == (50, 100)


class TestImageCropper:
    def test_order_quad_points(self) -> None:
        pts = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)
        ordered = _order_quad_points(pts)
        assert ordered.shape == (4, 2)
        assert ordered[0, 0] < ordered[2, 0]

    def test_order_quad_points_degenerate_all_same(self) -> None:
        pts = np.array([[50, 50], [50, 50], [50, 50], [50, 50]], dtype=np.float32)
        ordered = _order_quad_points(pts)
        assert ordered.shape == (4, 2)

    def test_order_quad_points_collinear(self) -> None:
        pts = np.array([[0, 0], [50, 0], [100, 0], [150, 0]], dtype=np.float32)
        ordered = _order_quad_points(pts)
        assert ordered.shape == (4, 2)

    @pytest.mark.parametrize(
        "polygon",
        [
            [(10, 10), (50, 10), (50, 50), (10, 50)],
            [(10, 10), (50, 15), (48, 50), (12, 48)],
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            [(10, 5), (60, 15), (50, 70), (0, 60)],
        ],
        ids=["rectangle", "skewed", "square", "rotated"],
    )
    def test_polygon_to_quad_returns_four_points(
        self, polygon: list[tuple[int, int]]
    ) -> None:
        quad = _polygon_to_quad(polygon)
        assert quad.shape == (4, 2)

    def test_perspective_transform_dimensions(self) -> None:
        pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        w, h, _ = _perspective_transform(pts)
        assert w == 100
        assert h == 50

    def test_perspective_transform_trapezoid(self) -> None:
        pts = np.array([[10, 0], [90, 0], [100, 50], [0, 50]], dtype=np.float32)
        w, h, _ = _perspective_transform(pts)
        assert w >= 90
        assert h == 50

    def test_cut_out_requires_four_or_more_points(self) -> None:
        img = Image.new("RGB", (100, 100), color="white")
        with pytest.raises(ValueError, match="Polygon must have 4 or more points"):
            ImageCropper.cut_out_image_by_polygon(img, [(10, 10), (20, 20)])

    @pytest.mark.parametrize(
        "polygon",
        [
            [(10, 10), (190, 10), (190, 190), (10, 190)],
            [
                (20, 20),
                (50, 20),
                (80, 20),
                (80, 50),
                (80, 80),
                (50, 80),
                (20, 80),
                (20, 50),
            ],
            [(25, 25), (75, 20), (80, 50), (70, 80), (30, 75), (20, 50)],
        ],
        ids=["rectangle", "many-points", "irregular-hexagon"],
    )
    def test_cut_out_returns_rgba_crop(self, polygon: list[tuple[int, int]]) -> None:
        img = Image.new("RGB", (300, 300), color="white")
        result = ImageCropper.cut_out_image_by_polygon(img, polygon)
        assert result.mode == "RGBA"
        assert result.size[0] > 0
        assert result.size[1] > 0
