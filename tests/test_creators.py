"""Tests for the Creators module."""

from pathlib import Path

import pytest
from PIL import Image

from digitex.creators.page_creator import PageDataCreator


class TestPageDataCreator:
    """Test suite for PageDataCreator class."""

    def test_page_data_creator_initialization(self) -> None:
        """Test that PageDataCreator stores image_size."""
        creator = PageDataCreator(image_size=640)
        assert creator.image_size == 640

    def test_create_resizes_images(self, tmp_path: Path) -> None:
        """Test that images are resized to train_image_size."""
        books_dir = tmp_path / "books"
        subject_dir = books_dir / "math" / "images" / "2024"
        subject_dir.mkdir(parents=True)
        img = Image.new("RGB", (400, 400), color="red")
        img.save(subject_dir / "page1.jpg")

        output_dir = tmp_path / "output"
        creator = PageDataCreator(image_size=100)
        creator.create(books_dir, output_dir, num_images=1)

        output_file = output_dir / "math_2024_page1.jpg"
        assert output_file.exists()
        result_img = Image.open(output_file)
        assert result_img.size == (100, 100)

    def test_create_raises_when_no_images(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised when no images found."""
        books_dir = tmp_path / "books"
        books_dir.mkdir()

        output_dir = tmp_path / "output"
        creator = PageDataCreator(image_size=100)
        with pytest.raises(FileNotFoundError):
            creator.create(books_dir, output_dir, num_images=1)

    def test_create_respects_num_images(self, tmp_path: Path) -> None:
        """Test that exactly num_images are created."""
        books_dir = tmp_path / "books"
        subject_dir = books_dir / "math" / "images" / "2024"
        subject_dir.mkdir(parents=True)

        for i in range(5):
            img = Image.new("RGB", (100, 100), color="red")
            img.save(subject_dir / f"page{i}.jpg")

        output_dir = tmp_path / "output"
        creator = PageDataCreator(image_size=100)
        creator.create(books_dir, output_dir, num_images=3)

        output_files = list(output_dir.glob("*.jpg"))
        assert len(output_files) == 3

    def test_create_converts_to_rgb(self, tmp_path: Path) -> None:
        """Test that RGBA images are converted to RGB."""
        books_dir = tmp_path / "books"
        subject_dir = books_dir / "math" / "images" / "2024"
        subject_dir.mkdir(parents=True)
        img = Image.new("RGBA", (100, 100), color="red")
        img.save(subject_dir / "page1.png")

        output_dir = tmp_path / "output"
        creator = PageDataCreator(image_size=100)
        creator.create(books_dir, output_dir, num_images=1)

        output_file = output_dir / "math_2024_page1.jpg"
        result_img = Image.open(output_file)
        assert result_img.mode == "RGB"

    def test_create_preserves_aspect_ratio(self, tmp_path: Path) -> None:
        """Test that aspect ratio is preserved when resizing."""
        books_dir = tmp_path / "books"
        subject_dir = books_dir / "math" / "images" / "2024"
        subject_dir.mkdir(parents=True)
        img = Image.new("RGB", (400, 200), color="red")
        img.save(subject_dir / "page1.jpg")

        output_dir = tmp_path / "output"
        creator = PageDataCreator(image_size=100)
        creator.create(books_dir, output_dir, num_images=1)

        output_file = output_dir / "math_2024_page1.jpg"
        result_img = Image.open(output_file)
        assert result_img.size == (100, 50)

    def test_create_output_dir_created(self, tmp_path: Path) -> None:
        """Test that output directory is created if it doesn't exist."""
        books_dir = tmp_path / "books"
        subject_dir = books_dir / "math" / "images" / "2024"
        subject_dir.mkdir(parents=True)
        img = Image.new("RGB", (100, 100), color="red")
        img.save(subject_dir / "page1.jpg")

        output_dir = tmp_path / "output" / "nested"
        creator = PageDataCreator(image_size=100)
        creator.create(books_dir, output_dir, num_images=1)

        assert output_dir.exists()
        assert len(list(output_dir.glob("*.jpg"))) == 1
