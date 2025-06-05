import os
import argparse

from digitex.training.superpoint.components.data import DatasetCreator
from digitex.training.superpoint.components.augmenter import KeypointAugmenter
from digitex.training.superpoint.components.visualizer import KeypointVisualizer


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

parser.add_argument(
    "--train_split", default=0.8, type=float, help="Split of train set."
)

parser.add_argument(
    "--num_keypoints", default=30, type=int, help="Number of keypoints per object."
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
NUM_KEYPOINTS = args.num_keypoints
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


def main() -> None:
    # Create dataset
    dataset_creator = DatasetCreator(
        raw_dir=RAW_DIR,
        dataset_dir=DATASET_DIR,
        num_keypoints=NUM_KEYPOINTS,
        train_split=TRAIN_SPLIT,
    )
    dataset_creator.create_dataset()

    # Augment dataset
    if AUGMENT:
        augmenter = KeypointAugmenter(raw_dir=RAW_DIR, dataset_dir=DATASET_DIR)
        augmenter.augment(num_images=AUG_IMAGES)

    # Visualize dataset
    if VISUALIZE:
        visualizer = KeypointVisualizer(
            dataset_dir=DATASET_DIR, check_images_dir=CHECK_IMAGES_DIR
        )
        visualizer.visualize(num_images=VIS_IMAGES)


if __name__ == "__main__":
    main()
