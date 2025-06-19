import os
import random
from PIL import Image

import numpy as np

from tqdm import tqdm


class MasksVisualizer:
    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
    ) -> None:
        """
        Initialize the visualizer.

        Args:
            dataset_dir: Path to the dataset directory containing train/val splits
            output_dir: Directory where visualization images will be saved
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir

        self._setup_dirs()

    def _setup_dirs(self) -> None:
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # Setup paths to train and validation directories
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")

    def _create_overlay_visualization(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Image.Image:
        """
        Create overlay visualization of mask on image.

        Args:
            image: Input image with shape (H, W, C) and values in [0, 1]
            mask: Segmentation mask with shape (H, W) and values in [0, 1]

        Returns:
            PIL Image with mask overlaid on image
        """
        # Create colored mask (red channel)
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 0] = mask

        # Convert to uint8
        image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        mask_uint8 = (colored_mask * 255).clip(0, 255).astype(np.uint8)

        # Create PIL images and blend
        image_pil = Image.fromarray(image_uint8)
        mask_pil = Image.fromarray(mask_uint8)

        overlay = Image.blend(
            image_pil.convert("RGBA"), mask_pil.convert("RGBA"), alpha=0.5
        )

        return overlay.convert("RGB")

    def _save_visualization(
        self, image: Image.Image, original_filename: str, split_name: str
    ) -> None:
        base_name = os.path.splitext(os.path.basename(original_filename))[0]
        output_filename = f"{base_name}_{split_name}.jpg"
        output_path = os.path.join(self.output_dir, output_filename)
        image.save(output_path)

    def _process_single_image(
        self, image_filename: str, images_dir: str, masks_dir: str, split_name: str
    ) -> None:
        # Setup paths
        base_name = os.path.splitext(image_filename)[0]
        image_path = os.path.join(images_dir, image_filename)
        mask_path = os.path.join(masks_dir, f"{base_name}.png")

        # Load and preprocess image
        image = np.array(Image.open(image_path)) / 255.0

        # Load and preprocess mask
        mask = np.array(Image.open(mask_path)) / 255.0

        # Skip if mask is empty
        if not np.any(mask > 0):
            return

        # Create and save visualization
        overlay = self._create_overlay_visualization(image, mask)
        self._save_visualization(overlay, image_filename, split_name)

    def _process_dataset_split(
        self, split_dir: str, split_name: str, num_images: int
    ) -> None:
        images_dir = os.path.join(split_dir, "images")
        masks_dir = os.path.join(split_dir, "masks")

        # Get and shuffle image list
        image_filenames = [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        random.shuffle(image_filenames)
        image_filenames = image_filenames[:num_images]

        # Process each image
        for image_filename in tqdm(
            image_filenames, desc=f"Visualizing {split_name} masks"
        ):
            self._process_single_image(
                image_filename, images_dir, masks_dir, split_name
            )

    def visualize(self, num_images: int = 10) -> None:
        for split_dir, split_name in [(self.train_dir, "train"), (self.val_dir, "val")]:
            if os.path.exists(split_dir):
                self._process_dataset_split(split_dir, split_name, num_images)
