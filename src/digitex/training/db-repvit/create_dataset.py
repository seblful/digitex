import os
import argparse

from components.data import DatasetCreator, DataChecker
from components.visualizer import Visualizer


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for augmented factor
parser.add_argument(
    "--train_split", default=0.8, type=float, help="Split of training dataset."
)


# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
TRAIN_SPLIT = args.train_split

HOME = os.getcwd()
DB_REPVIT_DIR = os.path.join(HOME, "src", "digitex", "training", "db-repvit")
DATA = os.path.join(DB_REPVIT_DIR, "data")
RAW_DIR = os.path.join(DATA, "raw-data")
DATASET_DIR = os.path.join(DATA, "dataset")
CHECK_IMAGES_DIR = os.path.join(DATA, "check-images")


def main() -> None:
    # Check transcription in data.json
    data_checker = DataChecker(raw_dir=RAW_DIR)
    data_checker.check()

    # Initializing dataset creator and process data
    dataset_creator = DatasetCreator(
        raw_dir=RAW_DIR, dataset_dir=DATASET_DIR, train_split=TRAIN_SPLIT
    )
    dataset_creator.create_dataset()

    # Visualize dataset annotations
    visualizer = Visualizer(dataset_dir=DATASET_DIR, check_images_dir=CHECK_IMAGES_DIR)
    visualizer.visualize(num_images=30)


if __name__ == "__main__":
    main()
