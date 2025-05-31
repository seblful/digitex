import os
import argparse


from digitex.training.svtr2.components.data import (
    SimpleDatasetCreator,
    LMDBDatasetCreator,
)
from digitex.training.svtr2.components.visualizer import (
    SimpleVisualizer,
    LMDBVisualizer,
)


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for raw data source
parser.add_argument(
    "--source",
    default="synth",
    type=str,
    choices=["ls", "synth"],
    help="Raw data source.",
)

# Get an arg for dataset type
parser.add_argument(
    "--dataset_type",
    default="lmdb",
    type=str,
    choices=["simple", "lmdb"],
    help="Type of dataset to create: 'simple' or 'lmdb'.",
)

# Get an arg for whether to use augmented images
parser.add_argument(
    "--use_aug",
    action="store_true",
    help="Whether to use augmented images. 'aug-images/0' and 'aug_gt.txt' should be located in raw-dir.",
)

# Get an arg for train split
parser.add_argument(
    "--train_split", default=0.9, type=float, help="Split of training dataset."
)

# Get an arg for word length
parser.add_argument(
    "--max_text_length", default=25, type=int, help="Max length of word in dataset."
)

# Get an arg for number of images to visualize
parser.add_argument(
    "--num_check_images", default=100, type=int, help="Number of images to visualize."
)


# Get our arguments from the parser
args = parser.parse_args()


HOME = os.getcwd()
SVTR_DIR = os.path.join(HOME, "src/digitex/training/svtr2")

SVTR_DATA_DIR = os.path.join(SVTR_DIR, "data")
SUBFOLDER = "finetune" if args.source == "ls" else "train"
RAW_DIR = os.path.join(SVTR_DATA_DIR, SUBFOLDER, "raw-data")
DATASET_DIR = os.path.join(SVTR_DATA_DIR, SUBFOLDER, "dataset")
CHECK_IMAGES_DIR = os.path.join(SVTR_DATA_DIR, SUBFOLDER, "check-images")

FONT_PATH = os.path.join(SVTR_DIR, "font", "Inter.ttf")


def main() -> None:
    # Choose dataset creator and visualizer based on dataset_type
    if args.dataset_type == "simple":
        dataset_creator_cls = SimpleDatasetCreator
        visualizer_cls = SimpleVisualizer
    else:
        dataset_creator_cls = LMDBDatasetCreator
        visualizer_cls = LMDBVisualizer

    # Initializing dataset creator and process data
    dataset_creator = dataset_creator_cls(
        raw_dir=RAW_DIR,
        dataset_dir=DATASET_DIR,
        train_split=args.train_split,
        max_text_length=args.max_text_length,
    )
    dataset_creator.create_dataset(source=args.source, use_aug=args.use_aug)

    # Visualize dataset annotations
    visualizer = visualizer_cls(
        dataset_dir=DATASET_DIR, check_images_dir=CHECK_IMAGES_DIR, font_path=FONT_PATH
    )
    visualizer.visualize(num_images=args.num_check_images)


if __name__ == "__main__":
    main()
