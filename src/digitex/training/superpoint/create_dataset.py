import os
import argparse
import yaml

from digitex.training.superpoint.components.data import DatasetCreator, HeatmapsCreator
from digitex.training.superpoint.components.augmenter import KeypointAugmenter
from digitex.training.superpoint.components.visualizer import (
    KeypointVisualizer,
    HeatmapsVisualizer,
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
SUPERPOINT_DIR = os.path.join(HOME, "src", "digitex", "training", "superpoint")

DATA_DIR = os.path.join(SUPERPOINT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw-data")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
CHECK_IMAGES_DIR = os.path.join(DATA_DIR, "check-images")

CONFIG_PATH = os.path.join(
    HOME, "src", "digitex", "training", "superpoint", "config", "config.yml"
)


def main() -> None:
    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Create dataset
    dataset_creator = DatasetCreator(
        raw_dir=RAW_DIR,
        dataset_dir=DATASET_DIR,
        max_keypoints=config["dataset"]["max_keypoints"],
        train_split=TRAIN_SPLIT,
    )
    dataset_creator.create_dataset()

    # Augment dataset
    if AUGMENT:
        augmenter = KeypointAugmenter(
            raw_dir=RAW_DIR,
            dataset_dir=DATASET_DIR,
        )
        augmenter.augment(num_images=AUG_IMAGES)

    heatmaps_creator = HeatmapsCreator(
        dataset_dir=DATASET_DIR,
        max_keypoints=config["dataset"]["max_keypoints"],
        heatmap_size=config["dataset"]["heatmap_size"],
        heatmap_sigma=config["dataset"]["heatmap_sigma"],
    )
    heatmaps_creator.create_heatmaps()

    # Visualize dataset
    if VISUALIZE:
        keypoint_visualizer = KeypointVisualizer(
            dataset_dir=DATASET_DIR, check_images_dir=CHECK_IMAGES_DIR
        )
        keypoint_visualizer.visualize(num_images=VIS_IMAGES)

        heatmaps_visualizer = HeatmapsVisualizer(
            dataset_dir=DATASET_DIR, check_images_dir=CHECK_IMAGES_DIR
        )
        heatmaps_visualizer.visualize(num_images=VIS_IMAGES)


if __name__ == "__main__":
    main()
