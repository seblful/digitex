"""Tests for the image processors: cropping, rotation, flattening, resizing."""

import numpy as np
import pytest
from PIL import Image

from digitex.domain.entities import PixelPolygon
from digitex.imaging import add_white_background, resize_image, rotate_image
from digitex.imaging.image import (
    ImageCropper,
    _order_quad_points,
    _perspective_transform,
    _polygon_to_quad,
)


class TestRotateImage:
    def test_ninety_degrees_swaps_the_dimensions(self) -> None:
        img = Image.new("RGB", (100, 50), color="white")

        result = rotate_image(img, 90.0)

        assert result.size == (50, 100)

    def test_the_canvas_grows_to_hold_the_rotated_image(self) -> None:
        """Nothing may be cut off — a tilted crop's corners stay inside."""
        img = Image.new("RGB", (100, 100), color="white")

        result = rotate_image(img, 45.0)

        assert result.size == (141, 141)


class TestAddWhiteBackground:
    def test_transparent_becomes_white(self) -> None:
        img = Image.new("RGBA", (10, 10), color=(100, 100, 100, 0))

        result = add_white_background(img)

        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (255, 255, 255)

    def test_opaque_unchanged(self) -> None:
        img = Image.new("RGBA", (10, 10), color=(50, 100, 150, 255))

        result = add_white_background(img)

        assert result.getpixel((0, 0)) == (50, 100, 150)

    def test_fully_transparent_image_is_all_white(self) -> None:
        img = Image.new("RGBA", (10, 10), color=(0, 0, 0, 0))
        result = add_white_background(img)
        assert result.mode == "RGB"
        assert result.size == (10, 10)
        assert result.getpixel((5, 5)) == (255, 255, 255)

    def test_partial_transparency_blends_toward_white(self) -> None:
        img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        result = add_white_background(img)
        pixel = result.getpixel((0, 0))
        # RGB, so a 3-tuple — narrowed because getpixel also describes the
        # single-band and out-of-bounds cases this image cannot produce.
        assert isinstance(pixel, tuple)
        red, green, blue = pixel
        assert red > 127
        assert green < 128
        assert blue < 128


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
        quad = _polygon_to_quad(PixelPolygon(polygon))
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
            ImageCropper.cut_out_image_by_polygon(
                img, PixelPolygon([(10, 10), (20, 20)])
            )

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
        result = ImageCropper.cut_out_image_by_polygon(img, PixelPolygon(polygon))
        assert result.mode == "RGBA"
        assert result.size[0] > 0
        assert result.size[1] > 0
