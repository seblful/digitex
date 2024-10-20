import os
import argparse

from components.dataset import DatasetCreator


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for task type
parser.add_argument("--task_type",
                    default="page",
                    type=str,
                    choices=["page", "question"],
                    help="Type of task type.")

# Get an arg for train split
parser.add_argument("--train_split",
                    default=0.8,
                    type=float,
                    help="Split of training dataset.")


# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
TASK_TYPE = args.task_type
TRAIN_SPLIT = args.train_split

HOME = os.getcwd()
DATA_DIR = os.path.join(HOME, "data", TASK_TYPE)
RAW_DIR = os.path.join(DATA_DIR, 'raw-data')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')


def main() -> None:
    # Initializing dataset creator and process data
    dataset_creator = DatasetCreator(RAW_DIR,
                                     DATASET_DIR,
                                     train_split=TRAIN_SPLIT)
    dataset_creator.process()


if __name__ == '__main__':
    main()
