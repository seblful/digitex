"""Shared utilities for extraction operations."""

import shutil
import tempfile
from contextlib import suppress
from pathlib import Path

import structlog

from digitex.core.corpus import is_image, question_image_number

logger = structlog.get_logger()


def numbered_images(folder: Path) -> list[tuple[int, Path]]:
    """Image files whose stem is a number, as ``(number, path)``, lowest first.

    A non-numeric stem is not a question image, so it is skipped with a warning.
    """
    images: list[tuple[int, Path]] = []
    for f in folder.iterdir():
        if not is_image(f):
            continue
        number = question_image_number(f)
        if number is None:
            logger.warning("Skipping file with non-numeric name", file_path=str(f))
            continue
        images.append((number, f))
    return sorted(images, key=lambda x: x[0])


def apply_renames(changes: list[tuple[Path, Path]]) -> None:
    """Apply ``(old, new)`` renames, staging the whole batch through a temp dir.

    A renumbering batch can map a file onto a name another file in the batch
    still holds, so every source is moved out before any target is written.
    Detouring one file at a time would not help — it lands on its final name
    before the next pair is read, leaving the collision intact. The staging dir
    sits beside the files so these stay renames rather than whole-file copies.

    Because every source leaves the folder before any target is written, a
    failure partway through would otherwise strand the batch in a staging
    directory that cleanup then deletes. Every move already made is unwound
    instead, so an interrupted batch leaves the folder as it was found.

    Args:
        changes: (old_path, new_path) pairs to apply.
    """
    if not changes:
        return

    tmp_dir = Path(tempfile.mkdtemp(dir=changes[0][0].parent))
    staged: list[tuple[Path, Path, Path]] = []
    applied: list[tuple[Path, Path]] = []
    try:
        for i, (old_path, new_path) in enumerate(changes):
            temp_path = tmp_dir / f"{i}_{new_path.name}"
            shutil.move(str(old_path), str(temp_path))
            staged.append((old_path, temp_path, new_path))

        for old_path, temp_path, new_path in staged:
            shutil.move(str(temp_path), str(new_path))
            applied.append((old_path, new_path))
    except BaseException:
        # Undo the finished moves first: that frees each original name before
        # the still-staged files are put back under it.
        for old_path, new_path in reversed(applied):
            with suppress(OSError):
                shutil.move(str(new_path), str(old_path))
        for old_path, temp_path, _ in staged:
            if temp_path.exists():
                with suppress(OSError):
                    shutil.move(str(temp_path), str(old_path))
        raise
    finally:
        # rmdir, not rmtree: anything still in here is a file that could not be
        # restored, and deleting it is exactly the loss this guards against.
        with suppress(OSError):
            tmp_dir.rmdir()


def renumber_folder_sequentially(
    folder: Path, dry_run: bool = True
) -> list[tuple[Path, Path]]:
    """Renumber images in a folder to fill gaps (1,2,4,5 → 1,2,3,4).

    Args:
        folder: Directory containing numbered image files.
        dry_run: If True, only preview changes without applying.

    Returns:
        List of (old_path, new_path) tuples for changed files.
    """
    images = numbered_images(folder)
    if not images:
        return []

    current_numbers = [n for n, _ in images]
    expected_numbers = list(range(1, len(images) + 1))

    if current_numbers == expected_numbers:
        return []

    changes: list[tuple[Path, Path]] = []
    for i, (_, old_path) in enumerate(images, 1):
        new_path = old_path.parent / f"{i}{old_path.suffix}"
        if old_path != new_path:
            changes.append((old_path, new_path))

    if not dry_run and changes:
        apply_renames(changes)

    return changes


def renumber_directory_tree(root: Path, dry_run: bool = True) -> int:
    """Renumber all image folders in a directory tree.

    Args:
        root: Root directory to search for image folders.
        dry_run: If True, only preview changes.

    Returns:
        Total number of files that were/would be renamed.
    """
    total = 0

    if not root.exists() or not root.is_dir():
        return 0

    def find_image_folders(current: Path) -> list[Path]:
        """Find every folder in the tree that directly holds images."""
        entries = sorted(current.iterdir())
        if any(is_image(item) for item in entries):
            return [current]

        folders: list[Path] = []
        for item in entries:
            if item.is_dir():
                folders.extend(find_image_folders(item))
        return folders

    for folder in find_image_folders(root):
        changes = renumber_folder_sequentially(folder, dry_run=dry_run)
        total += len(changes)

        if dry_run and changes:
            rel_path = folder.relative_to(root)
            logger.info(
                "Would renumber files",
                folder=str(rel_path),
                count=len(changes),
            )

    return total
