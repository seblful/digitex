import os

from modules.data_creator import DataCreator

# Paths
HOME = os.getcwd()
TESTING_DIR = os.path.dirname(os.path.dirname(HOME))

RAW_DATA_DIR = os.path.join(TESTING_DIR, "raw-data", "biology", "new")

WORDS_TRAIN_DIR = os.path.join(HOME, "data", "finetune", "train-data")

FAST_DIR = os.path.join(TESTING_DIR, "training", "fast")
PARTS_RAW_DIR = os.path.join(FAST_DIR, "data", "raw-data")
FAST_WORD_MODEL_PATH = os.path.join(FAST_DIR, "models", "fast_base2.pt")

YOLO_DIR = os.path.join(TESTING_DIR, "training", "yolo")
YOLO_PAGE_MODEL_PATH = os.path.join(YOLO_DIR, "models", "yolov11", "page_m.pt")
YOLO_QUESTION_MODEL_PATH = os.path.join(
    YOLO_DIR, "models", "yolov11", "question_x3.pt")


def main() -> None:
    # Create DataCreator instance
    data_creator = DataCreator()

    # Create parts data from question annotations
    data_creator.extract_words(parts_raw_dir=PARTS_RAW_DIR,
                               train_dir=WORDS_TRAIN_DIR,
                               num_images=100)

    # Create parts data from page and questions predictions
    data_creator.predict_words(raw_dir=RAW_DATA_DIR,
                               train_dir=WORDS_TRAIN_DIR,
                               yolo_page_model_path=YOLO_PAGE_MODEL_PATH,
                               yolo_question_model_path=YOLO_QUESTION_MODEL_PATH,
                               fast_word_model_path=FAST_WORD_MODEL_PATH,
                               scan_type="color",
                               num_images=1)


if __name__ == "__main__":
    main()
