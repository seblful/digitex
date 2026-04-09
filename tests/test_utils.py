"""Tests for the Utils module."""

from pathlib import Path

import pytest
from PIL import Image

from digitex.utils import (
    IMAGE_EXTENSIONS,
    _natural_sort_key,
    create_pdf_from_images,
    rename_images_to_sequential,
)


class TestNaturalSortKey:
    """Test suite for _natural_sort_key function."""

    def test_natural_sort_key_numeric(self) -> None:
        """Test that numeric parts are sorted correctly."""
        paths = [
            Path("Document_10.jpg"),
            Path("Document_2.jpg"),
            Path("Document_1.jpg"),
        ]
        sorted_paths = sorted(paths, key=_natural_sort_key)
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
        sorted_paths = sorted(paths, key=_natural_sort_key)
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
        sorted_paths = sorted(paths, key=_natural_sort_key)
        assert [p.name for p in sorted_paths] == [
            "Image_A.jpg",
            "Image_B.jpg",
            "Image_b.jpg",
        ]


class TestImageExtensions:
    """Test suite for IMAGE_EXTENSIONS constant."""

    def test_image_extensions_has_common_formats(self) -> None:
        """Test that common image extensions are present."""
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".bmp" in IMAGE_EXTENSIONS
        assert ".tiff" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS


class TestRenameImagesToSequential:
    """Test suite for rename_images_to_sequential function."""

    def test_rename_images_to_sequential(self, tmp_path: Path) -> None:
        """Test that images are renamed to sequential numbers."""
        folder = tmp_path / "subject1"
        folder.mkdir()
        (folder / "Document_10.jpg").touch()
        (folder / "Document_2.jpg").touch()
        (folder / "Document_1.jpg").touch()

        rename_images_to_sequential(tmp_path)

        files = sorted(folder.iterdir())
        assert [f.name for f in files] == ["1.jpg", "2.jpg", "3.jpg"]

    def test_rename_images_preserves_extension(self, tmp_path: Path) -> None:
        """Test that original extensions are preserved."""
        folder = tmp_path / "subject1"
        folder.mkdir()
        (folder / "image.png").touch()
        (folder / "photo.jpg").touch()

        rename_images_to_sequential(tmp_path)

        files = sorted(folder.iterdir())
        assert [f.name for f in files] == ["1.png", "2.jpg"]

    def test_rename_images_skips_non_image_files(self, tmp_path: Path) -> None:
        """Test that non-image files are not renamed."""
        folder = tmp_path / "subject1"
        folder.mkdir()
        (folder / "image.jpg").touch()
        (folder / "document.txt").touch()

        rename_images_to_sequential(tmp_path)

        files = sorted(folder.iterdir())
        assert [f.name for f in files] == ["1.jpg", "document.txt"]

    def test_rename_images_raises_for_missing_dir(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing directory."""
        missing_dir = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            rename_images_to_sequential(missing_dir)

    def test_rename_images_skips_empty_folders(self, tmp_path: Path) -> None:
        """Test that empty folders are skipped."""
        folder = tmp_path / "empty"
        folder.mkdir()
        (tmp_path / "subject").mkdir()

        rename_images_to_sequential(tmp_path)

        assert not list(folder.iterdir())


class TestCreatePdfFromImages:
    """Test suite for create_pdf_from_images function."""

    def test_create_pdf_from_images(self, tmp_path: Path) -> None:
        """Test that PDF is created from images."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        img1 = Image.new("RGB", (100, 100), color="red")
        img1.save(image_dir / "1.jpg")
        img2 = Image.new("RGB", (100, 100), color="blue")
        img2.save(image_dir / "2.jpg")

        output_path = tmp_path / "output.pdf"
        create_pdf_from_images(image_dir, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_create_pdf_raises_for_missing_dir(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing directory."""
        missing_dir = tmp_path / "nonexistent"
        output_path = tmp_path / "output.pdf"
        with pytest.raises(FileNotFoundError):
            create_pdf_from_images(missing_dir, output_path)

    def test_create_pdf_raises_for_no_images(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised when no images found."""
        image_dir = tmp_path / "empty"
        image_dir.mkdir()
        output_path = tmp_path / "output.pdf"
        with pytest.raises(FileNotFoundError):
            create_pdf_from_images(image_dir, output_path)

    def test_create_pdf_respects_max_dimensions(self, tmp_path: Path) -> None:
        """Test that images are resized according to max dimensions."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        img = Image.new("RGB", (400, 400), color="red")
        img.save(image_dir / "1.jpg")

        output_path = tmp_path / "output.pdf"
        create_pdf_from_images(image_dir, output_path, max_width=100, max_height=100)

        assert output_path.exists()
