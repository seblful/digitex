import os
import random
import shutil
from PIL import Image

import numpy as np

from tqdm import tqdm

from digitex.core.processors.file import FileProcessor

from .annotation import AnnotationCreator


class MaskGenerator:
    def __init__(self, mask_radius_ratio: float = 0.02) -> None:
        self.mask_radius_ratio = mask_radius_ratio

    def calculate_mask_radius(self, img_width: int, img_height: int) -> int:
        min_dim = min(img_height, img_width)
        radius = int(min_dim * self.mask_radius_ratio)
        return max(1, radius)

    def generate_mask_from_label(
        self,
        label: list[tuple[int, int, int]],
        img_width: int,
        img_height: int,
    ) -> np.ndarray:
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        if not label:
            return mask

        # Calculate mask radius for this image size
        mask_radius = self.calculate_mask_radius(img_width, img_height)

        # Create coordinate grids for efficient mask generation
        y_coords, x_coords = np.mgrid[0:img_height, 0:img_width]

        for lab in label:
            if not lab or len(lab) < 3:
                continue

            x, y, visibility = lab

            # Only process visible keypoints
            if visibility != 1:
                continue

            # Create circular mask around keypoint
            distance_squared = (x_coords - x) ** 2 + (y_coords - y) ** 2
            circle_mask = distance_squared <= mask_radius**2

            # Combine with existing mask using logical OR
            mask = np.logical_or(mask, circle_mask).astype(np.uint8)

        # Convert to 0-255 range for image saving
        return mask * 255


class DatasetCreator:
    def __init__(
        self,
        raw_dir: str,
        dataset_dir: str,
        train_split: float = 0.8,
        mask_radius_ratio: float = 0.02,
    ) -> None:
        """
        Initialize the dataset creator.

        Args:
            raw_dir: Path to raw data directory containing images and annotations
            dataset_dir: Path where the processed dataset will be created
            train_split: Fraction of data to use for training (0.0-1.0)
            mask_radius_ratio: Mask radius as fraction of image size (0.0-1.0)
        """
        # Input paths
        self.raw_dir = raw_dir
        self.raw_images_dir = os.path.join(raw_dir, "images")
        self.data_json_path = os.path.join(raw_dir, "data.json")
        self.annotations_json_path = os.path.join(raw_dir, "anns.json")

        # Output paths
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(dataset_dir, "train")
        self.val_dir = os.path.join(dataset_dir, "val")

        # Split configuration
        self.train_split = train_split
        self.val_split = 1 - train_split

        # Initialize components
        self.mask_generator = MaskGenerator(mask_radius_ratio=mask_radius_ratio)
        self.annotations_creator = AnnotationCreator(
            data_json_path=self.data_json_path,
            anns_json_path=self.annotations_json_path,
        )

        self._setup_dataset_directories()

    def _setup_dataset_directories(self) -> None:
        os.makedirs(self.dataset_dir, exist_ok=True)

        for split_dir in [self.train_dir, self.val_dir]:
            os.makedirs(split_dir, exist_ok=True)
            os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(split_dir, "masks"), exist_ok=True)

    def _create_train_val_split(self) -> tuple[list[str], list[str]]:
        image_filenames = [
            f
            for f in os.listdir(self.raw_images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        random.shuffle(image_filenames)

        # Calculate split sizes
        total_images = len(image_filenames)
        num_train = int(total_images * self.train_split)

        train_filenames = image_filenames[:num_train]
        val_filenames = image_filenames[num_train:]

        return train_filenames, val_filenames

    def _copy_image_to_split(self, split_dir: str, image_filename: str) -> None:
        source_path = os.path.join(self.raw_images_dir, image_filename)
        target_path = os.path.join(split_dir, "images", image_filename)
        shutil.copy2(source_path, target_path)

    def _get_image_dims(self, split_dir: str, image_filename: str) -> tuple[int, int]:
        image_path = os.path.join(split_dir, "images", image_filename)
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        return image_width, image_height

    def _create_and_save_mask(
        self,
        split_dir: str,
        keypoint_label: list[tuple[int, int, int]],
        image_filename: str,
    ) -> None:
        # Get image dimensions
        image_width, image_height = self._get_image_dims(split_dir, image_filename)

        # Generate mask using the mask generator
        mask = self.mask_generator.generate_mask_from_label(
            keypoint_label, image_width, image_height
        )

        # Always save mask (empty or with keypoints)
        self._save_mask_to_file(split_dir, image_filename, mask)

    def _save_mask_to_file(
        self, split_dir: str, image_filename: str, mask: np.ndarray
    ) -> None:
        base_name = os.path.splitext(image_filename)[0]
        mask_filename = f"{base_name}.png"
        mask_path = os.path.join(split_dir, "masks", mask_filename)

        mask_image = Image.fromarray(mask, mode="L")
        mask_image.save(mask_path)

    def _process_split_data(
        self,
        image_filenames: list[str],
        split_dir: str,
        annotations_dict: dict,
        split_name: str,
    ) -> None:
        for image_filename in tqdm(
            image_filenames, desc=f"Processing {split_name} data"
        ):
            # Copy image to split directory
            self._copy_image_to_split(split_dir, image_filename)

            # Create and save corresponding mask
            keypoint_label = annotations_dict[image_filename]
            self._create_and_save_mask(split_dir, keypoint_label, image_filename)

    def _partition_and_process_data(self) -> None:
        # Create train/validation split
        train_filenames, val_filenames = self._create_train_val_split()

        # Load all annotations
        annotations_dict = FileProcessor.read_json(json_path=self.annotations_json_path)

        # Process training data
        self._process_split_data(
            train_filenames, self.train_dir, annotations_dict, "train"
        )

        # Process validation data
        self._process_split_data(val_filenames, self.val_dir, annotations_dict, "val")

    def create_dataset(self) -> None:
        self.annotations_creator.create_annotations()
        self._partition_and_process_data()
