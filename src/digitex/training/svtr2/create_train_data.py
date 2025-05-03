import os

from digitex.core.creators.word import WordDataCreator

# Paths
HOME = os.getcwd()
BOOKS_DIR = os.path.join(HOME, "books")
PDF_DIR = os.path.join(BOOKS_DIR, "biology", "new")

SVTR_DIR = os.path.join(HOME, "src", "digitex", "training", "svtr2")
WORDS_TRAIN_DIR = os.path.join(SVTR_DIR, "data", "finetune", "train-data")

DB_REPVIT_DIR = os.path.join(HOME, "src", "digitex", "training", "db-repvit")
PARTS_RAW_DIR = os.path.join(DB_REPVIT_DIR, "data", "raw-data")

YOLO_PAGE_MODEL_PATH = os.path.join(HOME, "models", "page.pt")
YOLO_QUESTION_MODEL_PATH = os.path.join(HOME, "models", "question.pt")
DB_REPBIT_WORD_MODEL_PATH = os.path.join(HOME, "models", "word.pt")
DB_REPVIT_WORD_CONFIG_PATH = os.path.join(HOME, "models", "word_config.yml")


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
        db_repvit_word_model_path=DB_REPBIT_WORD_MODEL_PATH,
        db_repvit_word_config_path=DB_REPVIT_WORD_CONFIG_PATH,
        num_images=100,
    )


if __name__ == "__main__":
    main()
