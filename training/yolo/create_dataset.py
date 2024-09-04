import os
import argparse

from dataset import DatasetCreator


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for augmented factor
parser.add_argument("--train_split",
                    default=0.8,
                    type=float,
                    help="Split of training dataset.")


# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
TRAIN_SPLIT = args.train_split

HOME = os.getcwd()
DATA = os.path.join(HOME, "data")
RAW_DIR = os.path.join(DATA, 'raw-data')
DATASET_DIR = os.path.join(DATA, 'dataset')


def main() -> None:
    # Initializing dataset creator and process data
    dataset_creator = DatasetCreator(RAW_DIR,
                                     DATASET_DIR,
                                     train_split=TRAIN_SPLIT)
    dataset_creator.process()


if __name__ == '__main__':
    main()
