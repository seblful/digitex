import os
import argparse

import numpy as np

from components.dataset import DatasetCreator
from components.augmenter import Augmenter
from components.visualizer import Visualizer


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for task type
parser.add_argument("--data_subdir",
                    default="page",
                    type=str,
                    help="Type of task type.")

# Get an arg for train split
parser.add_argument("--train_split",
                    default=0.8,
                    type=float,
                    help="Split of training dataset.")

parser.add_argument("--augment",
                    action="store_true",
                    help="Whether to augment train data.")

parser.add_argument("--aug_factor",
                    default=3,
                    type=int,
                    help="How many augmented images to create.")

parser.add_argument("--visualize",
                    action="store_true",
                    help="Whether to visualize data.")


# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
DATA_SUBDIR = args.data_subdir
TRAIN_SPLIT = args.train_split
AUGMENT = args.augment
AUG_FACTOR = args.aug_factor
VISUALIZE = args.visualize

HOME = os.getcwd()
DATA_DIR = os.path.join(HOME, "data", DATA_SUBDIR)
RAW_DIR = os.path.join(DATA_DIR, 'raw-data')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
CHECK_IMAGES_DIR = os.path.join(DATA_DIR, "check-images")

# Overcome bool error related to new NumPy version
np.bool = np.bool_


def main() -> None:
    # Initializing dataset creator and process data
    dataset_creator = DatasetCreator(RAW_DIR,
                                     DATASET_DIR,
                                     train_split=TRAIN_SPLIT)
    dataset_creator.process()

    # Augment dataset
    if AUGMENT:
        augmenter = Augmenter(dataset_dir=DATASET_DIR,
                              aug_factor=AUG_FACTOR)
        augmenter.augment()

    if VISUALIZE:
        visualizer = Visualizer(dataset_dir=DATASET_DIR,
                                check_images_dir=CHECK_IMAGES_DIR)
        visualizer.visualize()


if __name__ == '__main__':
    main()
