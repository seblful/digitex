import os

from modules.data_creator import DataCreator
from modules.utils import create_pdf_from_images

# Paths
HOME = os.getcwd()

TESTING_DIR = os.path.dirname(os.path.dirname(HOME))
IMAGES_DIR = os.path.join(TESTING_DIR, "raw-data", "biology", "images")
RAW_DATA_DIR = os.path.join(TESTING_DIR, "raw-data", "biology", "new")

PAGE_RAW_DIR = os.path.join(HOME, "data", "page", "raw-data")
PAGE_TRAIN_DIR = os.path.join(HOME, "data", "page", "train-data")
PAGE_YOLO_PATH = os.path.join(HOME, "models", "page", "page_m.pt")


def main() -> None:
    # Create DataCreator instance
    data_creator = DataCreator()

    # # Create pdfs from images
    # for image_dir in os.listdir(IMAGES_DIR):
    #     image_dir = os.path.join(IMAGES_DIR, image_dir)
    #     create_pdf_from_images(image_dir=image_dir,
    #                            raw_dir=RAW_DATA_DIR)

    # Create data for page
    # data_creator.extract_pages(raw_dir=RAW_DATA_DIR,
    #                            train_dir=PAGE_TRAIN_DIR,
    #                            scan_type="color",
    #                            num_images=100)


if __name__ == "__main__":
    main()
