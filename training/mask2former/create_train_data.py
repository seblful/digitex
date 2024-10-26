import os

from modules.data_creator import DataCreator

# Paths
HOME = os.getcwd()

TESTING_DIR = os.path.dirname(os.path.dirname(HOME))

RAW_DATA_DIR = os.path.join(TESTING_DIR, "raw-data", "new")

YOLO_DIR = os.path.join(TESTING_DIR, "training", "yolo")
YOLO_PAGE_RAW_DIR = os.path.join(YOLO_DIR, "data", "page", "raw-data")
YOLO_PAGE_PATH = os.path.join(YOLO_DIR, "models", "yolov11", "page.pt")

TRAIN_DIR = os.path.join(HOME, "data", "train-data")


def main() -> None:
    # Create DataCreator instance
    data_creator = DataCreator()

    # # Create data for question from annotations
    # data_creator.create_question_train_data_raw(page_raw_dir=YOLO_PAGE_RAW_DIR,
    #                                             train_dir=TRAIN_DIR,
    #                                             num_images=100)

    # Create data for question from YOLO predictions
    data_creator.create_question_train_data_pred(raw_dir=RAW_DATA_DIR,
                                                 train_dir=TRAIN_DIR,
                                                 yolo_model_path=YOLO_PAGE_PATH,
                                                 num_images=100)


if __name__ == "__main__":
    main()
