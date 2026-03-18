from pathlib import Path

from digitex.core.page_creator import PageDataCreator
from digitex.utils import create_pdf_from_images

HOME = Path.cwd()
TRAINING_DIR = HOME / "training"
BOOKS_DIR = HOME / "books"
IMAGES_DIR = TRAINING_DIR / "data" / "page" / "books"
PAGE_TRAIN_DIR = TRAINING_DIR / "data" / "page" / "images"


def main() -> None:
    page_creator = PageDataCreator()

    # Create pdfs from images
    for image_dir in IMAGES_DIR.iterdir():
        create_pdf_from_images(image_dir=image_dir, raw_dir=BOOKS_DIR)

    # Create data for page
    page_creator.create(
        pdf_dir=BOOKS_DIR,
        output_dir=PAGE_TRAIN_DIR,
        num_images=100,
    )


if __name__ == "__main__":
    main()
