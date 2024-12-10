import os

from modules.data_creator import DataCreator

# Paths
HOME = os.getcwd()
TESTING_DIR = os.path.dirname(os.path.dirname(HOME))

PARTS_TRAIN_DIR = os.path.join(HOME, "data", "train-data")

YOLO_DIR = os.path.join(TESTING_DIR, "training", "yolo")
QUESTION_RAW_DIR = os.path.join(YOLO_DIR, "data", "question2", "raw-data")

YOLO_PAGE_PATH = os.path.join(YOLO_DIR, "models", "yolov11", "page_m.pt")
YOLO_QUESTION_PATH = os.path.join(
    YOLO_DIR, "models", "yolov11", "question_x3.pt")


def main() -> None:
    # Create DataCreator instance
    data_creator = DataCreator()

    # Create parts data from question annotations
    data_creator.extract_parts(question_raw_dir=QUESTION_RAW_DIR,
                               train_dir=PARTS_TRAIN_DIR,
                               num_images=100)


if __name__ == "__main__":
    main()
