import os
import argparse
import yaml

from digitex.training.segformer.components.data import DatasetCreator
from digitex.training.segformer.components.augmenter import MaskAugmenter
from digitex.training.segformer.components.visualizer import (
    MasksVisualizer,
)


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

parser.add_argument(
    "--train_split", default=0.8, type=float, help="Split of train set."
)

parser.add_argument(
    "--augment", action="store_true", help="Whether to augment train data."
)

parser.add_argument(
    "--aug_images", default=100, type=int, help="How many augmented images to create."
)

parser.add_argument(
    "--visualize", action="store_true", help="Whether to visualize data."
)

parser.add_argument(
    "--vis_images", default=50, type=int, help="How many images to visualize."
)

# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
TRAIN_SPLIT = args.train_split
AUGMENT = args.augment
AUG_IMAGES = args.aug_images
VISUALIZE = args.visualize
VIS_IMAGES = args.vis_images

HOME = os.getcwd()
SEGFORMER_DIR = os.path.join(HOME, "src", "digitex", "training", "segformer")

DATA_DIR = os.path.join(SEGFORMER_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw-data")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
CHECK_IMAGES_DIR = os.path.join(DATA_DIR, "check-images")

CONFIG_PATH = os.path.join(
    HOME, "src", "digitex", "training", "segformer", "config", "config.yml"
)


def main() -> None:
    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Create dataset
    dataset_creator = DatasetCreator(
        raw_dir=RAW_DIR,
        dataset_dir=DATASET_DIR,
        train_split=TRAIN_SPLIT,
        mask_radius_ratio=config["dataset"]["mask_radius_ratio"],
    )
    dataset_creator.create_dataset()

    # Augment dataset with mask-based augmentation
    if AUGMENT:
        augmenter = MaskAugmenter(
            raw_dir=RAW_DIR,
            dataset_dir=DATASET_DIR,
            mask_radius_ratio=config["dataset"]["mask_radius_ratio"],
        )
        augmenter.augment(num_images=AUG_IMAGES)

    # Visualize dataset
    if VISUALIZE:
        masks_visualizer = MasksVisualizer(
            dataset_dir=DATASET_DIR, output_dir=CHECK_IMAGES_DIR
        )
        masks_visualizer.visualize(num_images=VIS_IMAGES)


if __name__ == "__main__":
    main()
