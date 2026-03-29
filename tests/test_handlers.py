"""Tests for the Handlers module."""

from pathlib import Path

import pytest

from digitex.core.handlers import LabelHandler


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
