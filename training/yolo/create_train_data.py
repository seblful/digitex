import os

from modules.data_creator import DataCreator

# Paths
HOME = os.getcwd()

TESTING_DIR = os.path.dirname(os.path.dirname(HOME))
RAW_DIR = os.path.join(TESTING_DIR, "raw-data", "medium")
TRAIN_DIR = os.path.join(HOME, "data", "train-data")


def main() -> None:
    # Create DataCreator instance
    data_creator = DataCreator()

    # Create data
    data_creator.create_yolo_train_data(raw_dir=RAW_DIR,
                                        train_dir=TRAIN_DIR,
                                        scan_type="color",
                                        num_images=100)


if __name__ == "__main__":
    main()
