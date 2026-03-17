"""Pytest configuration and shared fixtures."""

import logging
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_image(tmp_path: Path) -> Image.Image:
    """Create a sample RGB image for testing.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        A sample PIL Image.
    """
    img = Image.new('RGB', (100, 100), color='red')
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)
    return img


@pytest.fixture
def sample_pdf_dir(tmp_path: Path) -> Path:
    """Create a directory with a sample PDF file.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        Path to the temporary directory.
    """
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    pdf_path = pdf_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test pdf\n%%EOF\n")

    return pdf_dir


@pytest.fixture
def sample_label_file(tmp_path: Path) -> Path:
    """Create a sample YOLO label file.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        Path to the label file.
    """
    label_dir = tmp_path / "labels"
    label_dir.mkdir(parents=True)

    label_path = label_dir / "image_0.txt"
    label_content = """0 0.5 0.5 0.8 0.8
1 0.2 0.2 0.3 0.3
"""
    label_path.write_text(label_content)

    return label_path


@pytest.fixture
def sample_classes_file(tmp_path: Path) -> Path:
    """Create a sample classes.txt file.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        Path to the classes file.
    """
    classes_path = tmp_path / "classes.txt"
    classes_content = """question
answer
"""
    classes_path.write_text(classes_content)

    return classes_path


@pytest.fixture
def mock_pdf_handler() -> Mock:
    """Create a mock PDFHandler for testing.

    Returns:
        Mocked PDFHandler instance.
    """
    return Mock()


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a sample dataset directory structure.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        Path to the dataset directory.
    """
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    classes_path = data_dir / "classes.txt"
    classes_path.write_text("question\nanswer\n")

    return data_dir
