"""Tests for the Utils module."""

from pathlib import Path

from digitex.utils import natural_sort_key


class TestNaturalSortKey:
    """Test suite for natural_sort_key function."""

    def test_natural_sort_key_numeric(self) -> None:
        """Test that numeric parts are sorted correctly."""
        paths = [
            Path("Document_10.jpg"),
            Path("Document_2.jpg"),
            Path("Document_1.jpg"),
        ]
        sorted_paths = sorted(paths, key=natural_sort_key)
        assert [p.name for p in sorted_paths] == [
            "Document_1.jpg",
            "Document_2.jpg",
            "Document_10.jpg",
        ]

    def test_natural_sort_key_mixed(self) -> None:
        """Test natural sort with mixed alphanumeric parts."""
        paths = [
            Path("page_20_image.png"),
            Path("page_3_image.png"),
            Path("page_10_image.png"),
        ]
        sorted_paths = sorted(paths, key=natural_sort_key)
        assert [p.name for p in sorted_paths] == [
            "page_3_image.png",
            "page_10_image.png",
            "page_20_image.png",
        ]

    def test_natural_sort_key_case_insensitive(self) -> None:
        """Test that sorting is case-insensitive."""
        paths = [
            Path("Image_B.jpg"),
            Path("Image_A.jpg"),
            Path("Image_b.jpg"),
        ]
        sorted_paths = sorted(paths, key=natural_sort_key)
        assert [p.name for p in sorted_paths] == [
            "Image_A.jpg",
            "Image_B.jpg",
            "Image_b.jpg",
        ]
