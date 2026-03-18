"""Tests for the DataCreator module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from digitex.core.extractor import DataCreator


class TestDataCreator:
    """Test suite for DataCreator class."""

    def test_init(self) -> None:
        """Test DataCreator initialization."""
        creator = DataCreator()
        assert creator.image_processor is not None
        assert creator.pdf_handler is not None
        assert creator.image_handler is not None
        assert creator.label_handler is not None

    def test_read_classes(self, sample_classes_file: Path) -> None:
        """Test reading classes from file."""
        classes_dict = DataCreator._read_classes(sample_classes_file)
        assert classes_dict == {0: "question", 1: "answer"}

    def test_save_image_creates_file(
        self,
        sample_image: Image.Image,
        tmp_path: Path,
    ) -> None:
        """Test that _save_image creates a file."""
        creator = DataCreator()
        train_dir = tmp_path / "train"
        train_dir.mkdir()

        num_saved = creator._save_image(
            0,
            train_dir=train_dir,
            image=sample_image,
            image_name="test.jpg",
            num_saved=0,
            num_images=1,
        )

        assert num_saved == 1
        assert (train_dir / "test_0.jpg").exists()

    def test_save_image_does_not_overwrite(
        self,
        sample_image: Image.Image,
        tmp_path: Path,
    ) -> None:
        """Test that _save_image doesn't overwrite existing files."""
        creator = DataCreator()
        train_dir = tmp_path / "train"
        train_dir.mkdir()

        num_saved = creator._save_image(
            0,
            train_dir=train_dir,
            image=sample_image,
            image_name="test.jpg",
            num_saved=0,
            num_images=1,
        )

        assert num_saved == 1

        num_saved = creator._save_image(
            0,
            train_dir=train_dir,
            image=sample_image,
            image_name="test.jpg",
            num_saved=num_saved,
            num_images=1,
        )

        assert num_saved == 1

    def test_extract_pages_invalid_scan_type(
        self,
        sample_pdf_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test extract_pages with invalid scan type."""
        creator = DataCreator()
        train_dir = tmp_path / "train"

        with pytest.raises(ValueError, match="Scan type must be one of"):
            creator.extract_pages(
                raw_dir=sample_pdf_dir,
                train_dir=train_dir,
                scan_type="invalid",
                num_images=1,
            )

    def test_extract_pages_no_pdfs(
        self,
        tmp_path: Path,
    ) -> None:
        """Test extract_pages with no PDF files."""
        creator = DataCreator()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        train_dir = tmp_path / "train"

        with pytest.raises(ValueError, match="No PDF files found"):
            creator.extract_pages(
                raw_dir=empty_dir,
                train_dir=train_dir,
                scan_type="color",
                num_images=1,
            )

    @patch('digitex.core.extractor.DataCreator._save_image')
    def test_extract_pages_max_attempts(
        self,
        mock_save: Mock,
        sample_pdf_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test extract_pages respects max_attempts limit."""
        mock_save.return_value = 0
        creator = DataCreator()
        train_dir = tmp_path / "train"

        with pytest.raises(ValueError, match="Failed to extract"):
            creator.extract_pages(
                raw_dir=sample_pdf_dir,
                train_dir=train_dir,
                scan_type="color",
                num_images=10,
                max_attempts=5,
            )

    def test_extract_questions_missing_directories(
        self,
        tmp_path: Path,
    ) -> None:
        """Test extract_questions with missing directories."""
        creator = DataCreator()
        missing_dir = tmp_path / "missing"
        train_dir = tmp_path / "train"

        with pytest.raises(ValueError, match="Images directory not found"):
            creator.extract_questions(
                page_raw_dir=missing_dir,
                train_dir=train_dir,
                num_images=1,
            )
