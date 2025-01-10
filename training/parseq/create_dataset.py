import os
import argparse

from components.data import DatasetCreator
from components.visualizer import Visualizer


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for raw data source
parser.add_argument("--source",
                    default="ls",
                    type=str,
                    choices=["ls", "synth"],
                    help="Raw data source.")

# Get an arg for train split
parser.add_argument("--train_split",
                    default=0.8,
                    type=float,
                    help="Split of training dataset.")

# Get an arg for map_size
parser.add_argument("--map_size",
                    default=100000000,
                    type=int,
                    help="Size of lmdb dataset in bytes.")


# Get our arguments from the parser
args = parser.parse_args()


HOME = os.getcwd()
DATA = os.path.join(HOME, "data")
SUBFOLDER = "finetune" if args.source == "ls" else "train"
RAW_DIR = os.path.join(DATA, SUBFOLDER, "raw-data")
DATASET_DIR = os.path.join(DATA, SUBFOLDER, "dataset")
CHECK_IMAGES_DIR = os.path.join(DATA, SUBFOLDER, "check-images")


def main() -> None:
    # Initializing dataset creator and process data
    dataset_creator = DatasetCreator(raw_dir=RAW_DIR,
                                     dataset_dir=DATASET_DIR,
                                     train_split=args.train_split,
                                     map_size=args.map_size)
    dataset_creator.create_dataset(source=args.source)

    # Visualize dataset annotations
    visualizer = Visualizer(dataset_dir=DATASET_DIR,
                            check_images_dir=CHECK_IMAGES_DIR,
                            source=args.source)
    visualizer.visualize(num_images=50)


if __name__ == '__main__':
    main()
