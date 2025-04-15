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

# Get an arg for whether to use augmented images
parser.add_argument("--use_aug",
                    action="store_true",
                    help="Whether to use augmented images. 'aug-images/0' and 'aug_gt.txt' should be located in raw-dir.")

# Get an arg for train split
parser.add_argument("--train_split",
                    default=0.8,
                    type=float,
                    help="Split of training dataset.")

# Get an arg for word length
parser.add_argument("--max_text_length",
                    default=31,
                    type=int,
                    help="Max length of word in dataset.")

# Get an arg for number of images to visualize
parser.add_argument("--num_check_images",
                    default=30,
                    type=int,
                    help="Number of images to visualize.")


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
                                     max_text_length=args.max_text_length)
    dataset_creator.create_dataset(source=args.source,
                                   use_aug=args.use_aug)

    # Visualize dataset annotations
    visualizer = Visualizer(dataset_dir=DATASET_DIR,
                            check_images_dir=CHECK_IMAGES_DIR)
    visualizer.visualize(num_images=args.num_check_images)


if __name__ == '__main__':
    main()
