"""Tests for the shared extraction utilities."""

from pathlib import Path

from digitex.extractors.utils import (
    renumber_directory_tree,
    renumber_folder_sequentially,
)


def _write_images(folder: Path, numbers: list[int]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for number in numbers:
        (folder / f"{number}.jpg").touch()


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


class TestRenumberFolderSequentially:
    def test_fills_gaps(self, tmp_path: Path) -> None:
        _write_images(tmp_path, [1, 2, 4, 5])

        changes = renumber_folder_sequentially(tmp_path, dry_run=False)

        assert len(changes) == 2
        assert _names(tmp_path) == ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]

    def test_already_sequential_is_a_no_op(self, tmp_path: Path) -> None:
        _write_images(tmp_path, [1, 2, 3])

        assert renumber_folder_sequentially(tmp_path, dry_run=False) == []

    def test_dry_run_reports_without_renaming(self, tmp_path: Path) -> None:
        _write_images(tmp_path, [1, 3])

        changes = renumber_folder_sequentially(tmp_path, dry_run=True)

        assert len(changes) == 1
        assert _names(tmp_path) == ["1.jpg", "3.jpg"]


class TestRenumberDirectoryTree:
    def test_renumbers_every_leaf_folder(self, tmp_path: Path) -> None:
        """Descending into one option/part folder must not skip its siblings."""
        leaves = {
            tmp_path / "2016" / "1" / "A": [1, 3],
            tmp_path / "2016" / "1" / "B": [2, 5],
            tmp_path / "2016" / "2" / "A": [1, 4],
            tmp_path / "2017" / "1" / "A": [7, 9],
        }
        for folder, numbers in leaves.items():
            _write_images(folder, numbers)

        total = renumber_directory_tree(tmp_path, dry_run=False)

        assert total == 6
        for folder in leaves:
            assert _names(folder) == ["1.jpg", "2.jpg"]

    def test_missing_root_is_zero(self, tmp_path: Path) -> None:
        assert renumber_directory_tree(tmp_path / "nope", dry_run=False) == 0
