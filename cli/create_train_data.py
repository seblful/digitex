from pathlib import Path

from digitex.core.page_creator import PageDataCreator
from digitex.utils import create_pdf_from_images

HOME = Path.cwd()
TESTING_DIR = HOME.parent.parent
IMAGES_DIR = TESTING_DIR / "books" / "biology" / "images"
RAW_DATA_DIR = TESTING_DIR / "books" / "biology" / "new"
PAGE_TRAIN_DIR = HOME / "data" / "page" / "train-data"


def main() -> None:
    page_creator = PageDataCreator()

    # Create pdfs from images
    for image_dir in IMAGES_DIR.iterdir():
        create_pdf_from_images(image_dir=image_dir, raw_dir=RAW_DATA_DIR)

    # Create data for page
    page_creator.create(
        pdf_dir=RAW_DATA_DIR,
        output_dir=PAGE_TRAIN_DIR,
        num_images=100,
    )


if __name__ == "__main__":
    main()
