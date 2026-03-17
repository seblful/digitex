"""Tests for the Processors module."""

from pathlib import Path

import pytest
import numpy as np
from PIL import Image

from modules.processors import FileProcessor, ImageProcessor


class TestImageProcessor:
    """Test suite for ImageProcessor class."""

    def test_init(self) -> None:
        """Test ImageProcessor initialization."""
        processor = ImageProcessor()
        assert processor.scan_types == ["bw", "gray", "color"]
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

    def test_process_invalid_scan_type(self) -> None:
        """Test process with invalid scan type."""
        processor = ImageProcessor()
        img = Image.new('RGB', (100, 100))

        with pytest.raises(ValueError, match="Scan type must be one of"):
            processor.process(img, scan_type="invalid")

    def test_process_valid_scan_type(self) -> None:
        """Test process with valid scan types."""
        processor = ImageProcessor()
        img = Image.new('RGB', (100, 100))

        for scan_type in processor.scan_types:
            result = processor.process(img, scan_type=scan_type)
            assert isinstance(result, Image.Image)
            assert result.mode == 'RGB'


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
