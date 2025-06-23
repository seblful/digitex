import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import SegformerImageProcessor


class SegFormerDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        model_name: str,
        image_size: tuple[int, int],
    ) -> None:
        """
        Dataset for loading images and corresponding segmentation masks for SegFormer.

        Args:
            dataset_dir: Path to the dataset directory containing images and masks.
            model_name: Name of the model to load the image processor from.
            image_size: Target size for images (height, width).
        """
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.masks_dir = os.path.join(dataset_dir, "masks")

        self.image_processor = SegformerImageProcessor.from_pretrained(model_name)
        self.image_size = image_size

        # Lists all images and masks files in the dataset directory
        self.images_listdir = sorted(
            [fn for fn in os.listdir(self.images_dir)],
            key=self._sort_key,
        )
        self.masks_listdir = sorted(
            [fn for fn in os.listdir(self.masks_dir)], key=self._sort_key
        )
        assert len(self.images_listdir) == len(self.masks_listdir), (
            "Number of images and masks must be the same."
        )

    def _sort_key(self, filename: str) -> str:
        return os.path.splitext(filename)[0]

    def _load_image(self, idx: int) -> Image.Image:
        image_filename = self.images_listdir[idx]
        image_path = os.path.join(self.images_dir, image_filename)

        image = Image.open(image_path).convert("RGB")
        return image

    def _load_mask(self, idx: int) -> Image.Image:
        mask_filename = self.masks_listdir[idx]
        mask_path = os.path.join(self.masks_dir, mask_filename)

        mask = Image.open(mask_path).convert("L")
        return mask

    def __len__(self) -> int:
        return len(self.images_listdir)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Load image and mask
        image = self._load_image(idx)
        mask = self._load_mask(idx)

        # Encode inputs
        encoded_inputs = self.image_processor(
            image,
            mask,
            do_resize=True,
            size={"height": self.image_size[0], "width": self.image_size[1]},
            do_normalize=True,
            do_reduce_labels=False,
            return_tensors="pt",
        )

        # Remove batch dimension
        for k in encoded_inputs.keys():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs
