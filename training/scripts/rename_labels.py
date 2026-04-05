"""Rename raw-data and page images/labels to structured format."""

from pathlib import Path

import structlog
import typer
from tqdm import tqdm

logger = structlog.get_logger()

SUBJECT_MAP = {"Биология": "biology"}


def rename_stem(stem: str) -> str | None:
    """Convert ЦТ_Биология_2016_medium_12 to biology_2016_12_medium.

    Args:
        stem: Original filename without extension.

    Returns:
        New filename stem, or None if format is unrecognized.
    """
    parts = stem.split("_")
    parts = [p for p in parts if p]

    if len(parts) < 4:
        return None

    subject = SUBJECT_MAP.get(parts[1])
    if subject is None:
        return None

    year = parts[2]
    page_type = parts[3]
    page_num = parts[4] if len(parts) > 4 else None

    if page_num is None:
        return None

    return f"{subject}_{year}_{page_num}_{page_type}"


def rename_page_stem(stem: str) -> str | None:
    """Convert 'ЦТ Биология  2014 old_0' to biology_2014_0_old.

    Handles spaces and double spaces in the original naming.

    Args:
        stem: Original filename without extension.

    Returns:
        New filename stem, or None if format is unrecognized.
    """
    parts = stem.split()
    parts = [p for p in parts if p]

    if len(parts) < 3:
        return None

    if parts[0] != "ЦТ":
        return None

    subject = SUBJECT_MAP.get(parts[1])
    if subject is None:
        return None

    year = parts[2]

    if len(parts) < 4:
        # Format: ЦТ Биология 2022_10 (no type)
        if "_" in year:
            year_parts = year.rsplit("_", 1)
            if len(year_parts) == 2 and year_parts[1].isdigit():
                return f"{subject}_{year_parts[0]}_{year_parts[1]}"
        return None

    rest = parts[3]

    if "_" in rest:
        rest_parts = rest.rsplit("_", 1)
        page_type = rest_parts[0]
        page_num = rest_parts[1]
        return f"{subject}_{year}_{page_num}_{page_type}"

    if len(parts) >= 5:
        page_type = rest
        page_num = parts[4]
        return f"{subject}_{year}_{page_num}_{page_type}"

    return None


def rename_dir(directory: Path, rename_fn, ext: str) -> tuple[int, int]:
    """Rename files in a directory using the given rename function.

    Args:
        directory: Directory containing files.
        rename_fn: Function that takes stem and returns new stem or None.
        ext: File extension including dot.

    Returns:
        Tuple of (renamed count, skipped count).
    """
    renamed = 0
    skipped = 0

    for file_path in tqdm(sorted(directory.glob(f"*{ext}")), desc=str(directory)):
        stem = file_path.stem

        # Strip hex prefix if present
        if (
            len(stem) > 8
            and stem[8] == "-"
            and all(c in "0123456789abcdef" for c in stem[:8])
        ):
            stem = stem[9:]

        new_stem = rename_fn(stem)
        if new_stem is None:
            logger.warning("unrecognized_format", name=file_path.name)
            skipped += 1
            continue

        new_path = directory / (new_stem + ext)
        if new_path.exists():
            logger.warning("target_exists", name=new_stem)
            skipped += 1
            continue

        file_path.rename(new_path)
        renamed += 1

    return renamed, skipped


def verify(base: Path) -> None:
    """Verify that images and labels are in sync across directories.

    Args:
        base: Training data base directory.
    """
    raw_images = {p.stem for p in (base / "raw-data" / "images").glob("*.jpg")}
    raw_labels = {p.stem for p in (base / "raw-data" / "labels").glob("*.txt")}
    page_images = {p.stem for p in (base / "images").glob("*.jpg")}

    images_no_label = raw_images - raw_labels
    labels_no_image = raw_labels - raw_images
    raw_no_page = raw_images - page_images
    page_no_raw = page_images - raw_images

    if images_no_label:
        logger.warning("images_without_labels", files=sorted(images_no_label))
    if labels_no_image:
        logger.warning("labels_without_images", files=sorted(labels_no_image))
    if raw_no_page:
        logger.warning("raw_without_page", files=sorted(raw_no_page))
    if page_no_raw:
        logger.warning("page_without_raw", files=sorted(page_no_raw))

    if not any([images_no_label, labels_no_image, raw_no_page, page_no_raw]):
        logger.info("verify_ok", count=len(raw_images))


def main() -> None:
    """Rename raw-data and page images/labels to structured format."""
    base = Path("training/data/page")

    total_renamed = 0
    total_skipped = 0

    # Rename raw-data labels
    r, s = rename_dir(base / "raw-data" / "labels", rename_stem, ".txt")
    total_renamed += r
    total_skipped += s

    # Rename raw-data images
    r, s = rename_dir(base / "raw-data" / "images", rename_stem, ".jpg")
    total_renamed += r
    total_skipped += s

    # Rename page/images
    r, s = rename_dir(base / "images", rename_page_stem, ".jpg")
    total_renamed += r
    total_skipped += s

    logger.info("done", renamed=total_renamed, skipped=total_skipped)
    verify(base)


if __name__ == "__main__":
    typer.run(main)
