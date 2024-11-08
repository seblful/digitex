import os

from modules.data_creator import DataCreator

# Paths
HOME = os.getcwd()

TESTING_DIR = os.path.dirname(os.path.dirname(HOME))

RAW_DATA_DIR = os.path.join(TESTING_DIR, "raw-data", "biology", "new")

YOLO_DIR = os.path.join(TESTING_DIR, "training", "yolo")
YOLO_PAGE_RAW_DIR = os.path.join(YOLO_DIR, "data", "page", "raw-data")
YOLO_QUESTION_RAW_DIR = os.path.join(YOLO_DIR, "data", "question", "raw-data")

YOLO_PAGE_PATH = os.path.join(YOLO_DIR, "models", "yolov11", "page_m.pt")
YOLO_QUESTION_PATH = os.path.join(
    YOLO_DIR, "models", "yolov11", "question_x2.pt")

TRAIN_DIR = os.path.join(HOME, "data", "train-data")


def main() -> None:
    # Create DataCreator instance
    data_creator = DataCreator()

    # # Create data from page annotations
    # target_classes = ["option",  "part"]
    # data_creator.create_ocr_train_data_raw(raw_dir=YOLO_PAGE_RAW_DIR,
    #                                        train_dir=TRAIN_DIR,
    #                                        target_classes=target_classes,
    #                                        num_images=1)

    # # Create data from question annotations
    # target_classes = ["answer", "number",
    #                   "option", "question", "spec", "table"]
    # data_creator.create_ocr_train_data_raw(raw_dir=YOLO_QUESTION_RAW_DIR,
    #                                        train_dir=TRAIN_DIR,
    #                                        target_classes=target_classes,
    #                                        num_images=1)

    # Create data from page predictions
    data_creator.create_ocr_train_data_pred(raw_dir=RAW_DATA_DIR,
                                            train_dir=TRAIN_DIR,
                                            yolo_page_model_path=YOLO_PAGE_PATH,
                                            yolo_question_model_path=YOLO_QUESTION_PATH,
                                            num_images=1)


if __name__ == "__main__":
    main()
