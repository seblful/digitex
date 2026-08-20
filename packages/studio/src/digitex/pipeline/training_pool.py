"""The pool of book pages copied out flat for annotation.

Annotation tools want one directory of images, not a corpus tree, so a page is
resized to the training size and renamed to carry the subject and year it came
from. Two ways in — sample a books tree (:meth:`PageDataCreator.create`) or take
a hand-written list (:meth:`PageDataCreator.add_from_file`) — and both funnel
into the same per-page save, because what a page is worth is the same question
either way.
"""

from __future__ import annotations

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
    """Copies book pages into the flat pool an annotation run reads."""

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
            # add_from_file's paths are typed by hand, so anything at all can
            # arrive here — including a directory, which passes exists().
            logger.warning("Skipping unrecognized book path", path=str(img_path))
            return PageOutcome.UNRECOGNIZED_PATH

        output_path = output_dir / training_page_name(subject, year, img_path.stem)
        if output_path.exists():
            return PageOutcome.ALREADY_PRESENT

        with Image.open(img_path) as source:
            # The pool is written as JPEG, which has no alpha channel to write.
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

        listed = paths_file.read_text(encoding="utf-8").strip().splitlines()
        if not listed:
            logger.warning("Paths file is empty")
            return

        found: list[Path] = []
        missing = 0
        for line in listed:
            entry = line.strip()
            # A blank line between two entries — the file's own ends were
            # already stripped off above.
            if not entry:
                continue
            path = Path(entry)
            if not path.exists():
                # One mistyped line must not cost the rest of the list.
                logger.warning("Source not found", path=str(path))
                missing += 1
                continue
            found.append(path)

        counts = self._save_images(found, output_dir, "Adding images")
        logger.info(
            "Done",
            processed=counts[PageOutcome.SAVED],
            skipped_exist=counts[PageOutcome.ALREADY_PRESENT],
            skipped_unrecognized=counts[PageOutcome.UNRECOGNIZED_PATH],
            skipped_missing=missing,
        )

    def create(
        self,
        books_dir: str | Path,
        output_dir: str | Path,
        num_images: int,
        subject: str | None = None,
    ) -> None:
        """Sample *num_images* pages, from *subject* alone or from all of them.

        Raises:
            FileNotFoundError: If there is no page to sample. Naming where it
                looked, because blaming the whole archive for one empty book
                sends the operator to the wrong place — and producing nothing
                quietly would read as "the pool is up to date".
        """
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
