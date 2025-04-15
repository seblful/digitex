import os

from digitex.core.creators.page import PageDataCreator
from digitex.core.utils import create_pdf_from_images

# Paths
HOME = os.getcwd()
BOOKS_DIR = os.path.join(HOME, "books")
IMAGES_DIR = os.path.join(BOOKS_DIR, "biology", "images")
RAW_DATA_DIR = os.path.join(BOOKS_DIR, "biology", "new")

PAGE_RAW_DIR = os.path.join(HOME, "data", "page", "raw-data")
PAGE_TRAIN_DIR = os.path.join(HOME, "data", "page", "train-data")
PAGE_YOLO_PATH = os.path.join(HOME, "models", "page", "page_m.pt")

QUESTION_RAW_DIR = os.path.join(HOME, "data", "question", "raw-data")
QUESTION_TRAIN_DIR = os.path.join(HOME, "data", "question", "train-data")
QUESTION_YOLO_PATH = os.path.join(HOME, "models", "question", "question_x3.pt")

TABLE_TRAIN_DIR = os.path.join(HOME, "data", "table", "train-data")


def main() -> None:
    # Create pdfs from images
    for image_dir in os.listdir(IMAGES_DIR):
        image_dir = os.path.join(IMAGES_DIR, image_dir)
        create_pdf_from_images(image_dir=image_dir, output_dir=RAW_DATA_DIR)
        break

    # # Create DataCreator instance
    # data_creator = PageDataCreator()
    # # Create data for page
    # data_creator.extract_pages(raw_dir=RAW_DATA_DIR,
    #                            train_dir=PAGE_TRAIN_DIR,
    #                            scan_type="color",
    #                            num_images=100)

    # # Create data for question from annotations
    # data_creator.extract_questions(page_raw_dir=PAGE_RAW_DIR,
    #                                train_dir=QUESTION_TRAIN_DIR,
    #                                num_images=100)

    # # Create data for question from YOLO predictions
    # data_creator.predict_questions(raw_dir=RAW_DATA_DIR,
    #                                train_dir=QUESTION_TRAIN_DIR,
    #                                yolo_model_path=PAGE_YOLO_PATH,
    #                                scan_type="color",
    #                                num_images=100)

    # # Create data for table from annotations
    # data_creator.extract_parts(
    #     question_raw_dir=QUESTION_RAW_DIR,
    #     train_dir=TABLE_TRAIN_DIR,
    #     num_images=80,
    #     target_classes=["table"],
    # )

    # # Create data for table from YOLO predictions
    # data_creator.predict_parts(
    #     raw_dir=RAW_DATA_DIR,
    #     train_dir=TABLE_TRAIN_DIR,
    #     yolo_page_model_path=PAGE_YOLO_PATH,
    #     yolo_question_model_path=QUESTION_YOLO_PATH,
    #     scan_type="color",
    #     num_images=80,
    #     target_classes=["table"],
    # )


if __name__ == "__main__":
    main()
