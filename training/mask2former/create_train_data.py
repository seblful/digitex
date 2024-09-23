import os

from modules.data_creator import DataCreator

# Paths
HOME = os.getcwd()

TESTING_DIR = os.path.dirname(os.path.dirname(HOME))

RAW_DIR = os.path.join(TESTING_DIR, "raw-data", "new")

YOLO_DIR = os.path.join(TESTING_DIR, "training", "yolo")
YOLO_RAW_DIR = os.path.join(YOLO_DIR, "data", "raw-data")
YOLO_MODEL_PATH = os.path.join(YOLO_DIR, "best.pt")

TRAIN_DIR = os.path.join(HOME, "data", "train-data")


def main() -> None:
    # Create DataCreator instance
    data_creator = DataCreator()

    # # Create data from yolo raw data
    # data_creator.create_mask2f_train_data_raw(yolo_raw_dir=YOLO_RAW_DIR,
    #                                           train_dir=TRAIN_DIR,
    #                                           num_images=100)

    # Create data from predictions
    data_creator.create_mask2f_train_data_pred(raw_dir=RAW_DIR,
                                               train_dir=TRAIN_DIR,
                                               yolo_model_path=YOLO_MODEL_PATH,
                                               num_images=100)


if __name__ == "__main__":
    main()
