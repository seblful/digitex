"""Page data creator for extracting images for training."""

import random
from collections import Counter
from enum import Enum, auto
from pathlib import Path

import structlog
from PIL import Image
from tqdm import tqdm

from digitex.domain.corpus import (
    PROCESSED,
    parse_book_page_path,
    training_page_name,
    walk_book_pages,
)
from digitex.imaging import resize_image

logger = structlog.get_logger()


class PageOutcome(Enum):
    """What happened to one book page offered to the training pool.

    ``UNRECOGNIZED_PATH`` and ``ALREADY_PRESENT`` are both "not saved" but mean
    opposite things to an operator, so they are counted apart.
    """

    SAVED = auto()
    ALREADY_PRESENT = auto()
    UNRECOGNIZED_PATH = auto()


class PageDataCreator:
    """Creator for preparing training images from book scans."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def _collect_images(self, books_dir: Path, subject: str | None) -> list[Path]:
        # Processed only: annotating a raw page would teach the model a
        # rendering it is never shown again.
        return list(walk_book_pages(books_dir, PROCESSED, subject))

    def _save_image(self, img_path: Path, output_dir: Path) -> PageOutcome:
        try:
            subject, year = parse_book_page_path(img_path)
        except ValueError:
            # Paths come from a user-supplied txt file in add_from_file.
            logger.warning("Skipping unrecognized book path", path=str(img_path))
            return PageOutcome.UNRECOGNIZED_PATH
        output_path = output_dir / training_page_name(subject, year, img_path.stem)
        if output_path.exists():
            return PageOutcome.ALREADY_PRESENT
        with Image.open(img_path) as source:
            image = source if source.mode == "RGB" else source.convert("RGB")
            resize_image(image, self.image_size, self.image_size).save(
                output_path, "JPEG"
            )
        return PageOutcome.SAVED

    def _save_images(
        self,
        paths: list[Path],
        output_dir: Path,
        desc: str,
    ) -> Counter[PageOutcome]:
        counts: Counter[PageOutcome] = Counter()
        for img_path in tqdm(paths, desc=desc):
            counts[self._save_image(img_path, output_dir)] += 1
        return counts

    def add_from_file(
        self,
        paths_file: str | Path,
        output_dir: str | Path,
    ) -> None:
        """Add images listed in a txt file to the output directory.

        Args:
            paths_file: Path to txt file with one relative image path per line.
            output_dir: Destination directory for processed images.
        """
        paths_file = Path(paths_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        lines = paths_file.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            logger.warning("Paths file is empty")
            return

        valid_paths: list[Path] = []
        skipped_missing = 0
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            img_path = Path(line)
            if not img_path.exists():
                logger.warning("Source not found", path=str(img_path))
                skipped_missing += 1
                continue
            valid_paths.append(img_path)

        counts = self._save_images(valid_paths, output_dir, "Adding images")
        logger.info(
            "Done",
            processed=counts[PageOutcome.SAVED],
            skipped_exist=counts[PageOutcome.ALREADY_PRESENT],
            skipped_unrecognized=counts[PageOutcome.UNRECOGNIZED_PATH],
            skipped_missing=skipped_missing,
        )

    def create(
        self,
        books_dir: str | Path,
        output_dir: str | Path,
        num_images: int,
        subject: str | None = None,
    ) -> None:
        """Sample *num_images* pages, from *subject* alone or from all of them."""
        books_dir = Path(books_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = self._collect_images(books_dir, subject)
        if not images:
            where = f"{books_dir}/{subject}" if subject else str(books_dir)
            raise FileNotFoundError(f"No images found in {where}")

        selected = random.sample(images, min(num_images, len(images)))
        logger.info(
            "Selected images",
            count=len(selected),
            books_dir=books_dir,
            subject=subject or "all",
        )

        counts = self._save_images(selected, output_dir, "Saving images")
        logger.info(
            "Saved images",
            saved=counts[PageOutcome.SAVED],
            skipped=counts[PageOutcome.ALREADY_PRESENT],
            skipped_unrecognized=counts[PageOutcome.UNRECOGNIZED_PATH],
            output_dir=output_dir,
        )
