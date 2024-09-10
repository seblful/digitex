import os

from modules.data_creator import DataCreator

# Paths
HOME = os.getcwd()

TESTING_DIR = os.path.dirname(os.path.dirname(HOME))
YOLO_RAW_DIR = os.path.join(TESTING_DIR, "training",
                            "yolo", "data", "raw-data")
TRAIN_DIR = os.path.join(HOME, "data", "train-data")


def main() -> None:
    # Create DataCreator instance
    data_creator = DataCreator()

    # Create data
    data_creator.create_mask2f_train_data(yolo_raw_dir=YOLO_RAW_DIR,
                                          train_dir=TRAIN_DIR,
                                          num_images=100)


if __name__ == "__main__":
    main()
