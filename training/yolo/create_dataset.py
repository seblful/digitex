import os
import argparse

from components.dataset import DatasetCreator
from components.augmenter import OBB_PolygonAugmenter, KeypointAugmenter
from components.visualizer import OBB_PolygonVisualizer, KeypointVisualizer


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

parser.add_argument("--data_subdir",
                    default="page",
                    type=str,
                    help="Type of task type.")

parser.add_argument("--train_split",
                    default=0.8,
                    type=float,
                    help="Split of train set.")

parser.add_argument("--anns_type",
                    default="keypoint",
                    type=str,
                    help="Annotation type, one of ['obb', 'polygon', 'keypoint']")

parser.add_argument("--num_keypoints",
                    default=30,
                    type=int,
                    help="Number of keypoints per object.")

parser.add_argument("--augment",
                    action="store_true",
                    help="Whether to augment train data.")

parser.add_argument("--aug_images",
                    default=100,
                    type=int,
                    help="How many augmented images to create.")

parser.add_argument("--visualize",
                    action="store_true",
                    help="Whether to visualize data.")

parser.add_argument("--vis_images",
                    default=50,
                    type=int,
                    help="How many images to visualize.")

# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
DATA_SUBDIR = args.data_subdir
TRAIN_SPLIT = args.train_split
ANNS_TYPE = args.anns_type
NUM_KEYPOINTS = args.num_keypoints
AUGMENT = args.augment
AUG_IMAGES = args.aug_images
VISUALIZE = args.visualize
VIS_IMAGES = args.vis_images

HOME = os.getcwd()
DATA_DIR = os.path.join(HOME, "data", DATA_SUBDIR)
RAW_DIR = os.path.join(DATA_DIR, 'raw-data')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
CHECK_IMAGES_DIR = os.path.join(DATA_DIR, "check-images")


def main() -> None:
    # Create dataset
    dataset_creator = DatasetCreator(raw_dir=RAW_DIR,
                                     dataset_dir=DATASET_DIR,
                                     num_keypoints=NUM_KEYPOINTS,
                                     train_split=TRAIN_SPLIT)
    dataset_creator.create(anns_type=ANNS_TYPE)

    # Augment dataset
    if AUGMENT:
        if ANNS_TYPE in ["obb", "polygon"]:
            augmenter = OBB_PolygonAugmenter(raw_dir=RAW_DIR,
                                             dataset_dir=DATASET_DIR,
                                             anns_type=ANNS_TYPE)
        elif ANNS_TYPE == "keypoint":
            augmenter = KeypointAugmenter(raw_dir=RAW_DIR,
                                          dataset_dir=DATASET_DIR,
                                          anns_type=ANNS_TYPE)
        augmenter.augment(num_images=AUG_IMAGES)

    # Visualize dataset
    if VISUALIZE:
        if ANNS_TYPE in ["obb", "polygon"]:
            visualizer = OBB_PolygonVisualizer(dataset_dir=DATASET_DIR,
                                               check_images_dir=CHECK_IMAGES_DIR,
                                               anns_type=ANNS_TYPE)
        elif ANNS_TYPE == "keypoint":
            visualizer = KeypointVisualizer(dataset_dir=DATASET_DIR,
                                            check_images_dir=CHECK_IMAGES_DIR,
                                            anns_type=ANNS_TYPE)
        visualizer.visualize(num_images=VIS_IMAGES)


if __name__ == '__main__':
    main()
