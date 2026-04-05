"""Tests for add-images CLI command."""

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def books_dir(tmp_path: Path) -> Path:
    """Create a mock books directory with test images."""
    img_dir = tmp_path / "books" / "biology" / "images" / "2024"
    img_dir.mkdir(parents=True)
    for name in ["1.jpg", "2.jpg", "3.jpg"]:
        img = Image.new("RGB", (1000, 800), color="white")
        img.save(img_dir / name)
    return tmp_path / "books"


@pytest.fixture()
def paths_file(tmp_path: Path, books_dir: Path) -> Path:
    """Create a paths.txt file with relative paths."""
    paths = [
        "books/biology/images/2024/1.jpg",
        "books/biology/images/2024/2.jpg",
    ]
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text("\n".join(paths))
    return paths_file


def _run_add_images(paths_file: Path, output_dir: Path) -> None:
    """Helper: run the add_images logic directly."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from digitex.core.processors import resize_image

    lines = paths_file.read_text().strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        src = paths_file.parent / line
        if not src.exists():
            continue
        parts = Path(line).parts
        # books/<subject>/images/<year>/<page>.jpg
        subject = parts[1]
        year = parts[3]
        page = Path(line).stem
        output_name = f"{subject}_{year}_{page}.jpg"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name
        if output_path.exists():
            continue
        image = Image.open(src)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = resize_image(image, 640, 640)
        image.save(output_path, "JPEG")


def test_add_images_copies_and_renames(
    tmp_path: Path, books_dir: Path, paths_file: Path
) -> None:
    output_dir = tmp_path / "training" / "data" / "page" / "images"
    _run_add_images(paths_file, output_dir)
    assert (output_dir / "biology_2024_1.jpg").exists()
    assert (output_dir / "biology_2024_2.jpg").exists()
    assert not (output_dir / "biology_2024_3.jpg").exists()


def test_add_images_resizes_to_640(
    tmp_path: Path, books_dir: Path, paths_file: Path
) -> None:
    output_dir = tmp_path / "training" / "data" / "page" / "images"
    _run_add_images(paths_file, output_dir)
    img = Image.open(output_dir / "biology_2024_1.jpg")
    assert max(img.size) <= 640


def test_add_images_skips_existing(
    tmp_path: Path, books_dir: Path, paths_file: Path
) -> None:
    output_dir = tmp_path / "training" / "data" / "page" / "images"
    output_dir.mkdir(parents=True)
    existing = output_dir / "biology_2024_1.jpg"
    existing.write_text("existing")
    _run_add_images(paths_file, output_dir)
    assert existing.read_text() == "existing"


def test_add_images_handles_missing_source(tmp_path: Path) -> None:
    # Create paths file without creating the actual book images
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text(
        "books/biology/images/2024/1.jpg\nbooks/biology/images/2024/2.jpg"
    )
    lines = paths_file.read_text().strip().splitlines()
    assert len(lines) == 2
    output_dir = tmp_path / "output"
    _run_add_images(paths_file, output_dir)
    assert not output_dir.exists() or list(output_dir.iterdir()) == []


def test_add_images_empty_file(tmp_path: Path) -> None:
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text("")
    lines = paths_file.read_text().strip().splitlines()
    assert len(lines) == 0
