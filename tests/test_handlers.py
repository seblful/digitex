"""Tests for the Handlers module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import numpy as np
from PIL import Image

from modules.handlers import ImageHandler, LabelHandler, PDFHandler


class TestImageHandler:
    """Test suite for ImageHandler class."""

    def test_crop_image(self) -> None:
        """Test cropping an image with polygon."""
        import numpy as np

        handler = ImageHandler()
        img = Image.new('RGB', (100, 100), color='red')

        points = [0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8]
        cropped = handler.crop_image(img, points, offset=0.0)

        assert isinstance(cropped, Image.Image)
        assert cropped.mode == 'RGB'
        assert cropped.size[0] < 100
        assert cropped.size[1] < 100

    def test_crop_image_invalid_points(self) -> None:
        """Test crop_image with invalid points."""
        handler = ImageHandler()
        img = Image.new('RGB', (100, 100), color='red')

        with pytest.raises(ValueError, match="Points list must contain an even number"):
            handler.crop_image(img, [0.1, 0.2, 0.3])

    def test_get_random_image_file_not_found(self, tmp_path: Path) -> None:
        """Test get_random_image with non-existent file."""
        handler = ImageHandler()

        with pytest.raises(FileNotFoundError):
            handler.get_random_image(["nonexistent.jpg"], str(tmp_path))


class TestLabelHandler:
    """Test suite for LabelHandler class."""

    def test_read_points(self, sample_label_file: Path) -> None:
        """Test reading points from label file."""
        handler = LabelHandler()
        points_dict = handler._read_points(sample_label_file)

        assert 0 in points_dict
        assert 1 in points_dict
        assert len(points_dict[0]) == 1
        assert len(points_dict[1]) == 1

    def test_get_random_label_file_not_found(self, tmp_path: Path) -> None:
        """Test get_random_label with non-existent label."""
        handler = LabelHandler()
        result = handler.get_random_label("nonexistent.jpg", tmp_path)

        assert result == (None, None)

    def test_get_random_label_found(self, sample_label_file: Path) -> None:
        """Test get_random_label with existing label."""
        handler = LabelHandler()
        labels_dir = sample_label_file.parent
        name, path = handler.get_random_label("image_0.jpg", labels_dir)

        assert name == "image_0.txt"
        assert path == str(sample_label_file)

    def test_points_to_abs_polygon(self) -> None:
        """Test converting normalized points to absolute coordinates."""
        handler = LabelHandler()
        points = [0.5, 0.5, 0.8, 0.8, 0.8, 0.5]
        abs_points = handler.points_to_abs_polygon(points, 100, 100)

        expected = [(50, 50), (80, 80), (80, 50)]
        assert abs_points == expected

    def test_points_to_abs_polygon_invalid(self) -> None:
        """Test points_to_abs_polygon with invalid points."""
        handler = LabelHandler()

        with pytest.raises(ValueError, match="Points list must contain an even number"):
            handler.points_to_abs_polygon([0.5, 0.5, 0.8], 100, 100)

    def test_get_points_no_labels(self, tmp_path: Path) -> None:
        """Test get_points when no labels exist."""
        handler = LabelHandler()
        result = handler.get_points(
            "nonexistent.jpg",
            tmp_path,
            {0: "test"},
            ["test"],
        )

        assert result == (-1, [])


class TestPDFHandler:
    """Test suite for PDFHandler class."""

    def test_get_page_image(self) -> None:
        """Test rendering a PDF page."""
        handler = PDFHandler()
        mock_page = Mock()
        mock_bitmap = Mock()
        mock_pil_image = Image.new('RGB', (100, 100))
        mock_bitmap.to_pil.return_value = mock_pil_image
        mock_page.render.return_value = mock_bitmap

        result = handler.get_page_image(mock_page, scale=3)

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'

    def test_get_random_image_file_not_found(self, tmp_path: Path) -> None:
        """Test get_random_image with non-existent PDF."""
        handler = PDFHandler()

        with pytest.raises(FileNotFoundError):
            handler.get_random_image(["nonexistent.pdf"], tmp_path)
