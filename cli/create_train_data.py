from pathlib import Path

from digitex.core.extractor import DataCreator
from digitex.utils import create_pdf_from_images

HOME = Path.cwd()
TESTING_DIR = HOME.parent.parent
IMAGES_DIR = TESTING_DIR / "raw-data" / "biology" / "images"
RAW_DATA_DIR = TESTING_DIR / "raw-data" / "biology" / "new"
PAGE_RAW_DIR = HOME / "data" / "page" / "raw-data"
PAGE_TRAIN_DIR = HOME / "data" / "page" / "train-data"
PAGE_YOLO_PATH = HOME / "models" / "page" / "page_m.pt"


def main() -> None:
    # Create DataCreator instance
    data_creator = DataCreator()

    # Create pdfs from images
    for image_dir in IMAGES_DIR.iterdir():
        create_pdf_from_images(image_dir=image_dir, raw_dir=RAW_DATA_DIR)

    # Create data for page
    data_creator.extract_pages(
        raw_dir=RAW_DATA_DIR,
        train_dir=PAGE_TRAIN_DIR,
        scan_type="color",
        num_images=100,
    )


if __name__ == "__main__":
    main()
