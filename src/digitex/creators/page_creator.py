"""Page data creator for extracting images for training."""

import random
from pathlib import Path

import structlog
from PIL import Image
from tqdm import tqdm

from digitex.core.processors import resize_image

logger = structlog.get_logger()


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
