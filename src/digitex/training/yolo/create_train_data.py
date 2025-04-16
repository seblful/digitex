import os

from tqdm import tqdm

from digitex.core.creators.page import PageDataCreator
from digitex.core.creators.question import QuestionDataCreator
from digitex.core.creators.part import PartDataCreator
from digitex.core.utils import create_pdf_from_images

# Paths
HOME = os.getcwd()
BOOKS_DIR = os.path.join(HOME, "books")
IMAGES_DIR = os.path.join(BOOKS_DIR, "biology", "images")
PDF_DIR = os.path.join(BOOKS_DIR, "biology", "new")

YOLO_DIR = os.path.join(HOME, "src", "digitex", "training", "yolo")
YOLO_DATA_DIR = os.path.join(YOLO_DIR, "data")

PAGE_RAW_DIR = os.path.join(YOLO_DATA_DIR, "page", "raw-data")
PAGE_TRAIN_DIR = os.path.join(YOLO_DATA_DIR, "page", "train-data")
PAGE_YOLO_PATH = os.path.join(HOME, "models", "page.pt")

QUESTION_RAW_DIR = os.path.join(YOLO_DATA_DIR, "question", "raw-data")
QUESTION_TRAIN_DIR = os.path.join(YOLO_DATA_DIR, "question", "train-data")
QUESTION_YOLO_PATH = os.path.join(HOME, "models", "question.pt")

TABLE_TRAIN_DIR = os.path.join(YOLO_DATA_DIR, "table-obb", "train-data")


def main() -> None:
    # # Create pdfs from images
    # for image_dir in tqdm(os.listdir(IMAGES_DIR), desc="Creating PDFs"):
    #     image_dir = os.path.join(IMAGES_DIR, image_dir)
    #     create_pdf_from_images(image_dir=image_dir, output_dir=PDF_DIR)

    # Create data for page from annotations
    data_creator = PageDataCreator()
    data_creator.extract(pdf_dir=PDF_DIR, train_dir=PAGE_TRAIN_DIR, num_images=100)

    # Create data for question from annotations
    data_creator = QuestionDataCreator()
    data_creator.extract(
        page_raw_dir=PAGE_RAW_DIR, train_dir=QUESTION_TRAIN_DIR, num_images=100
    )

    # Create data for question from YOLO predictions
    data_creator.predict(
        pdf_dir=PDF_DIR,
        train_dir=QUESTION_TRAIN_DIR,
        yolo_model_path=PAGE_YOLO_PATH,
        num_images=100,
    )

    # Create data for table from annotations
    data_creator = PartDataCreator()
    data_creator.extract(
        question_raw_dir=QUESTION_RAW_DIR,
        train_dir=TABLE_TRAIN_DIR,
        num_images=80,
        target_classes=["table"],
    )

    # Create data for table from YOLO predictions
    data_creator.predict(
        pdf_dir=PDF_DIR,
        train_dir=TABLE_TRAIN_DIR,
        yolo_page_model_path=PAGE_YOLO_PATH,
        yolo_question_model_path=QUESTION_YOLO_PATH,
        num_images=80,
        target_classes=["table"],
    )


if __name__ == "__main__":
    main()
