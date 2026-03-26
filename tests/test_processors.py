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
