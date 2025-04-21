import os

from digitex.core.creators.part import PartDataCreator

# Paths
HOME = os.getcwd()
BOOKS_DIR = os.path.join(HOME, "books")
PDF_DIR = os.path.join(BOOKS_DIR, "biology", "new")

DB_REPVIT_DIR = os.path.join(HOME, "src", "digitex", "training", "db-repvit")
PARTS_TRAIN_DIR = os.path.join(DB_REPVIT_DIR, "data", "train-data")

YOLO_DIR = os.path.join(HOME, "src", "digitex", "training", "yolo")
QUESTION_RAW_DIR = os.path.join(YOLO_DIR, "data", "question", "raw-data")

YOLO_PAGE_MODEL_PATH = os.path.join(HOME, "models", "page.pt")
YOLO_QUSTION_MODEL_PATH = os.path.join(HOME, "models", "question.pt")


def main() -> None:
    # Create DataCreator instance
    data_creator = PartDataCreator()

    # Create parts data from question annotations
    data_creator.extract(
        question_raw_dir=QUESTION_RAW_DIR, train_dir=PARTS_TRAIN_DIR, num_images=1
    )

    # Create parts data from page and questions predictions
    data_creator.predict(
        pdf_dir=PDF_DIR,
        train_dir=PARTS_TRAIN_DIR,
        yolo_page_model_path=YOLO_PAGE_MODEL_PATH,
        yolo_question_model_path=YOLO_QUSTION_MODEL_PATH,
        num_images=1,
    )


if __name__ == "__main__":
    main()
