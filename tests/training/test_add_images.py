"""Tests for add-images CLI command."""

import logging
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from digitex.config import settings as settings_module
from training.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_settings() -> None:
    """Reset the cached settings singleton between tests."""
    settings_module._settings = None


@pytest.fixture()
def books_dir(tmp_path: Path) -> Path:
    """Create a mock books directory with test images."""
    img_dir = tmp_path / "books" / "biology" / "images" / "2024"
    img_dir.mkdir(parents=True)
    for name in ["1.jpg", "2.jpg", "3.jpg"]:
        img = Image.new("RGB", (1000, 800), color="white")
        img.save(img_dir / name)
    return tmp_path / "books"


def _data_dir(tmp_path: Path) -> Path:
    return tmp_path / "training" / "data" / "page"


def _write_paths_file(tmp_path: Path, content: str) -> Path:
    data_dir = _data_dir(tmp_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths_file = data_dir / "paths.txt"
    paths_file.write_text(content)
    return paths_file


def test_add_images_copies_and_renames(
    tmp_path: Path, books_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_paths_file(
        tmp_path, "books/biology/images/2024/1.jpg\nbooks/biology/images/2024/2.jpg"
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["add-images"])
    assert result.exit_code == 0

    images_dir = _data_dir(tmp_path) / "images"
    assert (images_dir / "biology_2024_1.jpg").exists()
    assert (images_dir / "biology_2024_2.jpg").exists()
    assert not (images_dir / "biology_2024_3.jpg").exists()


def test_add_images_resizes_to_640(
    tmp_path: Path, books_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_paths_file(tmp_path, "books/biology/images/2024/1.jpg")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["add-images"])
    assert result.exit_code == 0

    output = _data_dir(tmp_path) / "images" / "biology_2024_1.jpg"
    img = Image.open(output)
    assert max(img.size) <= 640


def test_add_images_skips_existing(
    tmp_path: Path, books_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_paths_file(tmp_path, "books/biology/images/2024/1.jpg")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["add-images"])
    assert result.exit_code == 0

    images_dir = _data_dir(tmp_path) / "images"
    output = images_dir / "biology_2024_1.jpg"
    original_size = output.stat().st_size

    result = runner.invoke(app, ["add-images"])
    assert result.exit_code == 0
    assert output.stat().st_size == original_size
    assert len(list(images_dir.iterdir())) == 1


def test_add_images_handles_missing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _write_paths_file(
        tmp_path,
        "books/biology/images/2024/1.jpg\nbooks/biology/images/2024/2.jpg",
    )
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(app, ["add-images"])
    assert result.exit_code == 0
    assert "Source not found: books\\biology\\images\\2024\\1.jpg" in caplog.text
    assert "Source not found: books\\biology\\images\\2024\\2.jpg" in caplog.text


def test_add_images_empty_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_paths_file(tmp_path, "")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["add-images"])
    assert result.exit_code == 0
    assert "paths.txt is empty." in result.output


def test_add_images_no_paths_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["add-images"])
    assert result.exit_code == 1
    assert "paths.txt not found" in result.output
