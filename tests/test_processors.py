"""Tests for the Processors module."""

from pathlib import Path

import pytest
import numpy as np
from PIL import Image

from digitex.core.processors import (
    FileProcessor,
    SegmentProcessor,
    resize_img,
    resize_image,
)
from digitex.extractors.page_extractor import PageExtractor


class TestResizeImage:
    """Test suite for resize_img function (numpy arrays)."""

    def test_resize_img_smaller_than_max(self) -> None:
        """Test resize_img when image is already smaller than max_height."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        result = resize_img(img, 1000)

        assert result.shape == (500, 500, 3)

    def test_resize_img_larger_than_max(self) -> None:
        """Test resize_img when image is larger than max_height."""
        img = np.zeros((2000, 2000, 3), dtype=np.uint8)
        result = resize_img(img, 1000)

        assert result.shape[0] == 1000
        assert result.shape[1] == 1000

    def test_resize_image_pil_no_resize(self) -> None:
        """Test resize_image PIL without resizing."""
        img = Image.new("RGB", (100, 100))
        result = resize_image(img, 0)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_resize_image_pil_with_resize(self) -> None:
        """Test resize_image PIL with resizing."""
        img = Image.new("RGB", (2000, 2000))
        result = resize_image(img, 1000)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (1000, 1000)

    def test_remove_bg_returns_rgba(self) -> None:
        """Test remove_bg returns RGBA numpy array with transparent bright pixels."""
        processor = SegmentProcessor()
        img = np.ones((100, 100, 4), dtype=np.uint8) * 255
        img[20:80, 20:80, :3] = [50, 50, 50]
        result = processor.remove_bg(img)

        assert result.shape == (100, 100, 4)
        assert result.dtype == np.uint8
        alpha = result[:, :, 3]
        assert alpha[0, 0] == 0
        assert alpha[50, 50] == 255

    def test_remove_bg_invalid_threshold(self) -> None:
        """Test remove_bg raises ValueError on invalid threshold."""
        processor = SegmentProcessor()
        img = np.ones((100, 100, 4), dtype=np.uint8) * 255
        with pytest.raises(
            ValueError, match="Threshold must be an integer in range 0-255"
        ):
            processor.remove_bg(img, threshold=300)
        with pytest.raises(
            ValueError, match="Threshold must be an integer in range 0-255"
        ):
            processor.remove_bg(img, threshold=-1)


class TestFileProcessor:
    """Test suite for FileProcessor class."""

    def test_read_txt(self, tmp_path: Path) -> None:
        """Test reading a text file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line1\nline2\nline3\n")

        result = FileProcessor.read_txt(file_path)

        assert result == ["line1\n", "line2\n", "line3\n"]

    def test_write_txt(self, tmp_path: Path) -> None:
        """Test writing a text file."""
        file_path = tmp_path / "test.txt"
        lines = ["line1\n", "line2\n", "line3\n"]

        FileProcessor.write_txt(file_path, lines)

        assert file_path.read_text() == "line1\nline2\nline3\n"

    def test_read_json(self, tmp_path: Path) -> None:
        """Test reading a JSON file."""
        file_path = tmp_path / "test.json"
        file_path.write_text('{"key": "value"}')

        result = FileProcessor.read_json(file_path)

        assert result == {"key": "value"}

    def test_write_json(self, tmp_path: Path) -> None:
        """Test writing a JSON file."""
        file_path = tmp_path / "test.json"
        data = {"key": "value"}

        FileProcessor.write_json(data, file_path)

        assert file_path.exists()
        result = FileProcessor.read_json(file_path)
        assert result == data

    def test_write_yaml(self, tmp_path: Path) -> None:
        """Test writing a YAML file."""
        file_path = tmp_path / "test.yaml"
        data = {"key": "value"}

        FileProcessor.write_yaml(file_path, data)


class TestSegmentProcessor:
    """Test suite for SegmentProcessor."""

    def test_remove_bg(self) -> None:
        """Test remove_bg processes segment correctly."""
        processor = SegmentProcessor()
        img = np.ones((50, 50, 4), dtype=np.uint8) * 100

        result = processor.remove_bg(img, threshold=150)
        assert result.shape == (50, 50, 4)
        assert result.dtype == np.uint8

    def test_increase_darkness_darkens_midtones(self) -> None:
        """Test that increase_darkness darkens mid-tone pixels."""
        processor = SegmentProcessor()
        img = np.full((10, 10, 4), 128, dtype=np.uint8)
        img[:, :, 3] = 255

        result = processor.increase_darkness(img, gamma=0.8)

        assert result[0, 0, 0] < 128
        assert result[0, 0, 3] == 255

    def test_increase_darkness_gamma_one_unchanged(self) -> None:
        """Test that gamma=1.0 leaves image unchanged."""
        processor = SegmentProcessor()
        img = np.full((10, 10, 4), 100, dtype=np.uint8)
        img[:, :, 3] = 255

        result = processor.increase_darkness(img, gamma=1.0)

        np.testing.assert_array_equal(result, img)

    def test_increase_darkness_preserves_alpha(self) -> None:
        """Test that alpha channel is preserved."""
        processor = SegmentProcessor()
        img = np.full((10, 10, 4), 100, dtype=np.uint8)
        img[:, :, 3] = 128

        result = processor.increase_darkness(img, gamma=0.8)

        assert result[0, 0, 3] == 128

    def test_increase_darkness_invalid_gamma(self) -> None:
        """Test that invalid gamma raises ValueError."""
        processor = SegmentProcessor()
        img = np.full((10, 10, 4), 100, dtype=np.uint8)
        img[:, :, 3] = 255

        with pytest.raises(ValueError, match="gamma must be positive"):
            processor.increase_darkness(img, gamma=0)

        with pytest.raises(ValueError, match="gamma must be positive"):
            processor.increase_darkness(img, gamma=-1)

    def test_add_white_background_transparent_becomes_white(self) -> None:
        """Test that transparent areas become white."""
        processor = SegmentProcessor()
        img = np.full((10, 10, 4), 100, dtype=np.uint8)
        img[:, :, 3] = 0

        result = processor.add_white_background(img)

        assert result.shape[2] == 3
        assert result[0, 0, 0] == 255
        assert result[0, 0, 1] == 255
        assert result[0, 0, 2] == 255

    def test_add_white_background_opaque_unchanged(self) -> None:
        """Test that opaque pixels remain unchanged."""
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

    def test_add_white_background_returns_rgb(self) -> None:
        """Test that result is 3-channel RGB."""
        processor = SegmentProcessor()
        img = np.full((10, 10, 4), 100, dtype=np.uint8)
        img[:, :, 3] = 128

        result = processor.add_white_background(img)

        assert result.shape[2] == 3


def test_crop_and_save_threshold_forces_png(tmp_path: Path) -> None:
    """Test that threshold preprocess mode saves as PNG regardless of image_format."""
    from unittest.mock import patch

    from PIL import Image as PILImage

    extractor = PageExtractor(
        model_path=Path("dummy.pt"),
        render_scale=2,
        image_format="jpg",
    )

    image = PILImage.new("RGB", (200, 200), color="white")
    polygon = [(10, 10), (190, 10), (190, 190), (10, 190)]
    output_path = tmp_path / "output.png"

    with patch.object(SegmentProcessor, "process") as mock_process:
        mock_process.return_value = PILImage.new("RGB", (180, 180), color="white")
        extractor._crop_and_save(image, polygon, output_path)
        mock_process.assert_called_once()

    assert output_path.exists()
    assert output_path.suffix == ".png"

    saved = PILImage.open(output_path)
    assert saved.mode == "RGB"
