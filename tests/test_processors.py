"""Tests for the Processors module."""

from pathlib import Path

import pytest
import numpy as np
from PIL import Image

from digitex.core.processors import FileProcessor, ImageProcessor
from digitex.utils import prepare_image


class TestImageProcessor:
    """Test suite for ImageProcessor class."""

    def test_init(self) -> None:
        """Test ImageProcessor initialization."""
        processor = ImageProcessor()
        assert processor.lower_blue is not None
        assert processor.upper_blue is not None

    def test_resize_image_smaller_than_max(self) -> None:
        """Test resize_image when image is already smaller than max_height."""
        processor = ImageProcessor()
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        result = processor.resize_image(img, max_height=1000)

        assert result.shape == (500, 500, 3)

    def test_resize_image_larger_than_max(self) -> None:
        """Test resize_image when image is larger than max_height."""
        processor = ImageProcessor()
        img = np.zeros((2000, 2000, 3), dtype=np.uint8)
        result = processor.resize_image(img, max_height=1000)

        assert result.shape[0] == 1000
        assert result.shape[1] == 1000

    def test_illuminate_image(self) -> None:
        """Test illuminate_image."""
        processor = ImageProcessor()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = processor.illuminate_image(img, alpha=2.0, beta=10)

        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_prepare_image_no_resize(self) -> None:
        """Test prepare_image without resizing."""
        img = Image.new('RGB', (100, 100))
        result = prepare_image(img, max_height=0)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'
        assert result.size == (100, 100)

    def test_prepare_image_with_resize(self) -> None:
        """Test prepare_image with resizing."""
        img = Image.new('RGB', (2000, 2000))
        result = prepare_image(img, max_height=1000)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'
        assert result.size == (1000, 1000)

    def test_remove_bg_grabcut_returns_bgra(self) -> None:
        """Test remove_bg_grabcut returns 4-channel BGRA image with non-trivial alpha."""
        processor = ImageProcessor()
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img[20:80, 20:80] = [50, 50, 50]
        result = processor.remove_bg_grabcut(img)

        assert result.shape == (100, 100, 4)
        assert result.dtype == np.uint8
        alpha = result[:, :, 3]
        assert not np.all(alpha == 0)
        assert not np.all(alpha == 255)

    def test_remove_bg_grabcut_empty_image(self) -> None:
        """Test remove_bg_grabcut raises ValueError on empty image."""
        processor = ImageProcessor()
        img = np.array([], dtype=np.uint8).reshape(0, 0, 3)

        with pytest.raises(ValueError, match="Image is empty"):
            processor.remove_bg_grabcut(img)

    def test_remove_bg_grabcut_1x1_image(self) -> None:
        """Test remove_bg_grabcut raises ValueError on 1x1 image."""
        processor = ImageProcessor()
        img = np.ones((1, 1, 3), dtype=np.uint8) * 255

        with pytest.raises(ValueError, match="Image must be larger than 2x2"):
            processor.remove_bg_grabcut(img)

    def test_remove_bg_grabcut_2x2_image(self) -> None:
        """Test remove_bg_grabcut raises ValueError on 2x2 image."""
        processor = ImageProcessor()
        img = np.ones((2, 2, 3), dtype=np.uint8) * 255

        with pytest.raises(ValueError, match="Image must be larger than 2x2"):
            processor.remove_bg_grabcut(img)

    def test_remove_bg_grabcut_float32_image(self) -> None:
        """Test remove_bg_grabcut raises ValueError on float32 image."""
        processor = ImageProcessor()
        img = np.ones((100, 100, 3), dtype=np.float32) * 255

        with pytest.raises(ValueError, match="Image must have dtype uint8"):
            processor.remove_bg_grabcut(img)

    def test_remove_bg_grabcut_wrong_channels(self) -> None:
        """Test remove_bg_grabcut raises ValueError on wrong number of channels."""
        processor = ImageProcessor()
        img = np.ones((100, 100, 4), dtype=np.uint8)

        with pytest.raises(ValueError, match="Image must have 3 channels"):
            processor.remove_bg_grabcut(img)

    def test_remove_bg_grabcut_grayscale(self) -> None:
        """Test remove_bg_grabcut raises ValueError on grayscale image."""
        processor = ImageProcessor()
        img = np.ones((100, 100), dtype=np.uint8) * 255

        with pytest.raises(ValueError, match="Image must have 3 channels"):
            processor.remove_bg_grabcut(img)


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

        assert file_path.exists()
