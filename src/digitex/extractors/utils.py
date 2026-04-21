"""Shared utilities for extraction operations."""

import shutil
import tempfile
from pathlib import Path

import structlog

logger = structlog.get_logger()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def find_image_files(directory: Path, recursive: bool = True) -> list[Path]:
    """Find all image files in a directory.

    Args:
        directory: Directory to search for images.
        recursive: If True, search subdirectories recursively.

    Returns:
        List of image file paths sorted by name.
    """
    if not directory.exists() or not directory.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    images = [
        p
        for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda p: p.name)


def count_images_by_hierarchy(
    root: Path,
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    """Count images organized by subject/year/option/part hierarchy.

    Args:
        root: Root directory containing the hierarchy.

    Returns:
        Nested dictionary: {subject: {year: {option: {part: count}}}}
    """

    def count_in_folder(folder: Path) -> int:
        return sum(
            1
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    result: dict[str, dict[str, dict[str, dict[str, int]]]] = {}

    if not root.exists() or not root.is_dir():
        return result

    for subject_dir in root.iterdir():
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name

        for year_dir in subject_dir.iterdir():
            if not year_dir.is_dir():
                continue
            year = year_dir.name

            for option_dir in year_dir.iterdir():
                if not option_dir.is_dir():
                    continue
                option = option_dir.name

                for part_dir in option_dir.iterdir():
                    if not part_dir.is_dir():
                        continue
                    part = part_dir.name
                    count = count_in_folder(part_dir)

                    result.setdefault(subject, {}).setdefault(year, {}).setdefault(
                        option, {}
                    )[part] = count

    return result


def count_subject_images(
    subject_dir: Path,
) -> dict[str, dict[str, dict[str, int]]]:
    """Count images in a subject directory by year/option/part.

    Args:
        subject_dir: Subject directory containing year folders.

    Returns:
        Nested dictionary: {year: {option: {part: count}}}
    """
    def count_in_folder(folder: Path) -> int:
        return sum(
            1
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    result: dict[str, dict[str, dict[str, int]]] = {}

    if not subject_dir.exists() or not subject_dir.is_dir():
        return result

    for year_dir in subject_dir.iterdir():
        if not year_dir.is_dir():
            continue
        year = year_dir.name

        for option_dir in year_dir.iterdir():
            if not option_dir.is_dir():
                continue
            option = option_dir.name

            for part_dir in option_dir.iterdir():
                if not part_dir.is_dir():
                    continue
                part = part_dir.name
                count = count_in_folder(part_dir)

                result.setdefault(year, {}).setdefault(option, {})[part] = count

    return result


def get_mode_values(values: list[int]) -> set[int]:
    """Get the mode(s) of a list of integers.

    Args:
        values: List of integers.

    Returns:
        Set of most frequent values.
    """
    from collections import Counter

    if not values:
        return set()

    counter = Counter(values)
    max_count = counter.most_common(1)[0][1]
    return {v for v, c in counter.items() if c == max_count}


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
    images = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                num = int(f.stem)
                images.append((num, f))
            except ValueError:
                logger.warning(
                    "Skipping file with non-numeric name", file_path=str(f)
                )

    if not images:
        return []

    images.sort(key=lambda x: x[0])
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
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for old_path, new_path in changes:
                temp_path = tmp_dir / new_path.name
                shutil.move(str(old_path), str(temp_path))
                shutil.move(str(temp_path), str(new_path))

    return changes


def renumber_directory_tree(
    root: Path, subject: str | None = None, dry_run: bool = True
) -> int:
    """Renumber all image folders in a directory tree.

    Args:
        root: Root directory to search for image folders.
        subject: Optional subject filter.
        dry_run: If True, only preview changes.

    Returns:
        Total number of files that were/would be renamed.
    """
    total = 0

    if not root.exists() or not root.is_dir():
        return 0

    def find_image_folders(current: Path) -> list[Path]:
        """Find folders that contain images."""
        for item in current.iterdir():
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
                return [current]
            if item.is_dir():
                if subject and current.parent.name != subject:
                    continue
                return find_image_folders(item)
        return []

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


def count_total_images(root: Path) -> tuple[int, int]:
    """Count total images and folders.

    Args:
        root: Root directory to count images in.

    Returns:
        Tuple of (total_images, total_folders).
    """
    total_images = 0
    total_folders = 0

    if not root.exists() or not root.is_dir():
        return 0, 0

    for item in root.rglob("*"):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            total_images += 1
        elif item.is_dir():
            has_images = any(
                p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                for p in item.iterdir()
            )
            if has_images:
                total_folders += 1

    return total_images, total_folders
