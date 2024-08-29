import os
import random

from data_creator import DataCreator

# Paths
HOME = os.getcwd()

TESTING_DIR = os.path.dirname(HOME)
RAW_DIR = os.path.join(TESTING_DIR, "raw-data")
TRAIN_DIR = os.path.join(HOME, "data", "train-data")


def main() -> None:
    # Create DataCreator instance
    data_creator = DataCreator(raw_data_dir=RAW_DIR,
                               train_dir=TRAIN_DIR)

    # Create data
    data_creator.create_train_data(num_images=20)


if __name__ == "__main__":
    main()
