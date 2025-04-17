import os

from digitex.core.creators.word import WordDataCreator

# Paths
HOME = os.getcwd()
BOOKS_DIR = os.path.join(HOME, "books")
PDF_DIR = os.path.join(BOOKS_DIR, "biology", "new")

SVTR_DIR = os.path.join(HOME, "src", "digitex", "training", "svtr")
WORDS_TRAIN_DIR = os.path.join(SVTR_DIR, "data", "finetune", "train-data")

FAST_DIR = os.path.join(HOME, "src", "digitex", "training", "fast")
PARTS_RAW_DIR = os.path.join(FAST_DIR, "data", "raw-data")

YOLO_PAGE_MODEL_PATH = os.path.join(HOME, "models", "page.pt")
YOLO_QUESTION_MODEL_PATH = os.path.join(HOME, "models", "question.pt")
FAST_WORD_MODEL_PATH = os.path.join(HOME, "models", "word.pt")


def main() -> None:
    # Create DataCreator instance
    data_creator = WordDataCreator()

    # Create parts data from question annotations
    data_creator.extract(
        parts_raw_dir=PARTS_RAW_DIR, train_dir=WORDS_TRAIN_DIR, num_images=100
    )

    # Create parts data from page and questions predictions
    data_creator.predict(
        pdf_dir=PDF_DIR,
        train_dir=WORDS_TRAIN_DIR,
        yolo_page_model_path=YOLO_PAGE_MODEL_PATH,
        yolo_question_model_path=YOLO_QUESTION_MODEL_PATH,
        fast_word_model_path=FAST_WORD_MODEL_PATH,
        num_images=100,
    )


if __name__ == "__main__":
    main()
