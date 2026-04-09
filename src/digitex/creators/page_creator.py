"""Page data creator for extracting images for training."""

import logging
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from digitex.core.processors import resize_image

logger = logging.getLogger(__name__)


class PageDataCreator:
    """Creator for preparing training images from book scans."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def _parse_book_path(self, img_path: Path) -> tuple[str, str]:
        """Extract subject and year from a book image path.

        Expected structure: books/<subject>/images/<year>/<page>.<ext>

        Returns:
            Tuple of (subject, year).
        """
        parts = img_path.parts
        year_idx = parts.index("images") + 1
        subject_idx = parts.index("books") + 1
        return parts[subject_idx], parts[year_idx]

    def _collect_images(self, books_dir: Path) -> list[Path]:
        from digitex.utils import IMAGE_EXTENSIONS

        images: list[Path] = []
        for img_path in books_dir.rglob("*"):
            if (
                img_path.is_file()
                and img_path.suffix.lower() in IMAGE_EXTENSIONS
                and "images" in img_path.parts
            ):
                images.append(img_path)
        return images

    def _save_image(self, img_path: Path, output_dir: Path) -> bool:
        subject, year = self._parse_book_path(img_path)
        output_path = output_dir / f"{subject}_{year}_{img_path.stem}.jpg"
        if output_path.exists():
            return False
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = resize_image(image, self.image_size, self.image_size)
        image.save(output_path, "JPEG")
        return True

    def _save_images(
        self,
        paths: list[Path],
        output_dir: Path,
        desc: str,
    ) -> tuple[int, int]:
        saved = 0
        skipped = 0
        for img_path in tqdm(paths, desc=desc):
            if self._save_image(img_path, output_dir):
                saved += 1
            else:
                skipped += 1
        return saved, skipped

    def create(
        self,
        books_dir: str | Path,
        output_dir: str | Path,
        num_images: int,
    ) -> None:
        books_dir = Path(books_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = self._collect_images(books_dir)
        if not images:
            raise FileNotFoundError(f"No images found in {books_dir}")

        selected = random.sample(images, min(num_images, len(images)))
        logger.info(f"Selected {len(selected)} images from {books_dir}")

        saved, skipped = self._save_images(selected, output_dir, "Saving images")
        logger.info(
            f"Saved {saved} images, skipped {skipped} (already exist) to {output_dir}"
        )

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

        lines = paths_file.read_text().strip().splitlines()
        if not lines:
            logger.warning("Paths file is empty.")
            return

        valid_paths: list[Path] = []
        skipped_missing = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            img_path = Path(line)
            if not img_path.exists():
                logger.warning(f"Source not found: {img_path}")
                skipped_missing += 1
                continue
            valid_paths.append(img_path)

        saved, skipped_exist = self._save_images(
            valid_paths, output_dir, "Adding images"
        )
        logger.info(
            f"Done. Processed: {saved}, Skipped (exist): {skipped_exist}, "
            f"Skipped (missing): {skipped_missing}"
        )
