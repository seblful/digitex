"""Tests for the shared extraction utilities."""

import shutil
from pathlib import Path

import pytest

from digitex.extractors import utils
from digitex.extractors.utils import (
    apply_renames,
    renumber_directory_tree,
    renumber_folder_sequentially,
)


def _write_images(folder: Path, numbers: list[int]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for number in numbers:
        (folder / f"{number}.jpg").touch()


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


class TestApplyRenames:
    """The staging round-trip and what it guarantees when a move fails.

    Every source leaves the folder before any target lands, so a failure
    partway through must put them all back rather than lose them.
    """

    @pytest.mark.parametrize("fail_on_call", [4, 5, 6])
    def test_a_failed_move_restores_every_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_on_call: int
    ) -> None:
        for number in (1, 2, 3):
            (tmp_path / f"{number}.jpg").write_text(str(number))
        changes = [
            (tmp_path / f"{n}.jpg", tmp_path / f"{n + 1}.jpg") for n in (3, 2, 1)
        ]

        real_move = shutil.move
        calls = 0

        def flaky_move(src: str, dst: str) -> object:
            nonlocal calls
            calls += 1
            # Calls 1-3 stage the batch out; 4 onwards write the targets.
            if calls == fail_on_call:
                raise OSError("target busy")
            return real_move(src, dst)

        monkeypatch.setattr(utils.shutil, "move", flaky_move)

        with pytest.raises(OSError, match="target busy"):
            apply_renames(changes)

        assert _names(tmp_path) == ["1.jpg", "2.jpg", "3.jpg"]
        for number in (1, 2, 3):
            assert (tmp_path / f"{number}.jpg").read_text() == str(number)

    def test_applies_a_shift_that_collides_within_the_batch(
        self, tmp_path: Path
    ) -> None:
        for number in (1, 2, 3):
            (tmp_path / f"{number}.jpg").write_text(str(number))

        apply_renames(
            [(tmp_path / f"{n}.jpg", tmp_path / f"{n + 1}.jpg") for n in (3, 2, 1)]
        )

        assert _names(tmp_path) == ["2.jpg", "3.jpg", "4.jpg"]
        assert (tmp_path / "2.jpg").read_text() == "1"
        assert (tmp_path / "4.jpg").read_text() == "3"


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
