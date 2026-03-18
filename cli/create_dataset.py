import argparse
from pathlib import Path

from digitex.ml.yolo.augmenter import PolygonAugmenter
from digitex.ml.yolo.dataset import DatasetCreator
from digitex.ml.yolo.visualizer import PolygonVisualizer


parser = argparse.ArgumentParser(description="Get some hyperparameters.")

parser.add_argument(
    "--data_subdir",
    default="page",
    type=str,
    help="Type of task type.",
)

parser.add_argument("--train_split", default=0.8, type=float, help="Split of train set.")

parser.add_argument(
    "--anns_type",
    default="polygon",
    type=str,
    help="Annotation type, one of ['polygon']",
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

args = parser.parse_args()

DATA_SUBDIR = args.data_subdir
TRAIN_SPLIT = args.train_split
ANNS_TYPE = args.anns_type
AUGMENT = args.augment
AUG_IMAGES = args.aug_images
VISUALIZE = args.visualize
VIS_IMAGES = args.vis_images

HOME = Path.cwd()
DATA_DIR = HOME / "data" / DATA_SUBDIR
RAW_DIR = DATA_DIR / "books"
DATASET_DIR = DATA_DIR / "dataset"
CHECK_IMAGES_DIR = DATA_DIR / "check-images"


def main() -> None:
    dataset_creator = DatasetCreator(
        raw_dir=RAW_DIR,
        dataset_dir=DATASET_DIR,
        train_split=TRAIN_SPLIT,
    )
    dataset_creator.create(anns_type=ANNS_TYPE)

    if AUGMENT:
        augmenter = PolygonAugmenter(
            raw_dir=str(RAW_DIR), dataset_dir=str(DATASET_DIR)
        )
        augmenter.augment(num_images=AUG_IMAGES)

    if VISUALIZE:
        visualizer = PolygonVisualizer(
            dataset_dir=str(DATASET_DIR),
            check_images_dir=str(CHECK_IMAGES_DIR),
        )
        visualizer.visualize(num_images=VIS_IMAGES)


if __name__ == "__main__":
    main()
