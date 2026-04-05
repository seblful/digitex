"""Page data creator for extracting random images for training."""

import logging
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from digitex.core.processors import resize_image

logger = logging.getLogger(__name__)


class PageDataCreator:
    """Creator for selecting and saving random images for training data."""

    def __init__(self, train_image_size: int) -> None:
        self.train_image_size = train_image_size

    def _collect_images(self, books_dir: Path) -> list[Path]:
        from digitex.utils import IMAGE_EXTENSIONS

        images: list[Path] = []
        for subject_dir in books_dir.iterdir():
            if not subject_dir.is_dir():
                continue
            images_dir = subject_dir / "images"
            if not images_dir.exists():
                continue
            for year_dir in images_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                for img_path in year_dir.iterdir():
                    if (
                        img_path.is_file()
                        and img_path.suffix.lower() in IMAGE_EXTENSIONS
                    ):
                        images.append(img_path)
        return images

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

        skipped = 0
        saved = 0
        for img_path in tqdm(selected, desc="Saving images"):
            book_name = img_path.parent.parent.parent.name
            year = img_path.parent.name
            output_path = output_dir / f"{book_name}_{year}_{img_path.stem}.jpg"
            if output_path.exists():
                skipped += 1
                continue
            image = Image.open(img_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = resize_image(image, self.train_image_size, self.train_image_size)
            image.save(output_path, "JPEG")
            saved += 1

        logger.info(
            f"Saved {saved} images, skipped {skipped} (already exist) to {output_dir}"
        )
