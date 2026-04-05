"""Tests for add-images CLI command."""

import logging
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from training.cli import app

runner = CliRunner()


@pytest.fixture()
def books_dir(tmp_path: Path) -> Path:
    """Create a mock books directory with test images."""
    img_dir = tmp_path / "books" / "biology" / "images" / "2024"
    img_dir.mkdir(parents=True)
    for name in ["1.jpg", "2.jpg", "3.jpg"]:
        img = Image.new("RGB", (1000, 800), color="white")
        img.save(img_dir / name)
    return tmp_path / "books"


def test_add_images_copies_and_renames(
    tmp_path: Path, books_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text(
        "books/biology/images/2024/1.jpg\nbooks/biology/images/2024/2.jpg"
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["add-images"])
    assert result.exit_code == 0


def test_add_images_resizes_to_640(
    tmp_path: Path, books_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text("books/biology/images/2024/1.jpg")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["add-images"])
    assert result.exit_code == 0


def test_add_images_skips_existing(
    tmp_path: Path, books_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text("books/biology/images/2024/1.jpg")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["add-images"])
    assert result.exit_code == 0


def test_add_images_handles_missing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text(
        "books/biology/images/2024/1.jpg\nbooks/biology/images/2024/2.jpg"
    )
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.DEBUG):
        result = runner.invoke(app, ["add-images"])
    assert result.exit_code == 0
    assert "Skipped (missing): 2" in caplog.text


def test_add_images_empty_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text("")
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
