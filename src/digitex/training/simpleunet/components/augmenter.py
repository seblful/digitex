import os
from PIL import Image

import numpy as np
import albumentations as A
from tqdm import tqdm

from digitex.core.utils import get_random_img


class BaseAugmenter:
    def __init__(
        self,
        raw_dir: str,
        dataset_dir: str,
    ) -> None:
        # Paths
        self.raw_dir = raw_dir
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.images_dir = os.path.join(self.train_dir, "images")

        self.img_ext = ".jpg"
        self.anns_ext = ".txt"

        self._transforms = None
        self._augmenter = None

    @property
    def transforms(self) -> A.Compose:
        if self._transforms is None:
            self._transforms = [
                A.AdditiveNoise(p=0.3),
                A.Downscale(scale_range=[0.4, 0.9], p=0.3),
                A.RGBShift(p=0.3),
                A.RingingOvershoot(p=0.3),
                A.Spatter(mean=[0.5, 0.6], p=0.2),
                A.ToGray(p=0.4),
                A.ChannelShuffle(p=0.3),
                A.Emboss(p=0.3),
                A.GaussNoise(std_range=[0.05, 0.15], p=0.3),
                A.HueSaturationValue(p=0.3),
                A.MedianBlur(p=0.3),
                A.PlanckianJitter(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomShadow(shadow_intensity_range=[0.1, 0.4], p=0.3),
                A.SaltAndPepper(amount=[0.01, 0.03], p=0.2),
                A.GaussianBlur(blur_limit=6, p=0.3),
                A.ISONoise(p=0.2),
                A.MotionBlur(p=0.3),
                A.PlasmaBrightnessContrast(p=0.3),
                A.RandomFog(p=0.3),
                A.Sharpen(p=0.4),
                A.Blur(p=0.3),
                A.Illumination(p=0.3),
                A.CLAHE(p=0.3),
                A.Posterize(p=0.3),
                A.Affine(scale=[0.92, 1.08], fill=255, p=0.4),
                A.CoarseDropout(fill=255, p=0.1),
                A.Pad(padding=[15, 15], fill=255, p=0.4),
                A.RandomScale(p=0.4),
                A.SafeRotate(limit=(-3, 3), fill=255, p=0.4),
            ]

        return self._transforms

    @property
    def augmenter(self) -> None:
        return self._augmenter

    @augmenter.setter
    def augmenter(self, value) -> None:
        self._augmenter = value

    def find_path(self, img_path: str) -> str:
        name = os.path.splitext(img_path)[0]

        increment = 1
        while True:
            filename = f"{name}_aug_{increment}{self.img_ext}"
            filepath = os.path.join(self.train_dir, filename)
            if not os.path.exists(filepath):
                return filename
            increment += 1

    def transform_and_save_image(self, img_path: str, img: np.ndarray) -> None:
        aug_img_filename = self.find_path(img_path)
        aug_img_path = os.path.join(self.images_dir, aug_img_filename)

        # Save image
        image = Image.fromarray(img)
        image.save(aug_img_path)

        return aug_img_filename


class MaskAugmenter(BaseAugmenter):
    def __init__(self, raw_dir: str, dataset_dir: str) -> None:
        super().__init__(raw_dir, dataset_dir)
        self.masks_dir = os.path.join(self.train_dir, "masks")

    @property
    def augmenter(self) -> A.Compose:
        if self._augmenter is None:
            self._augmenter = A.Compose(self.transforms)
        return self._augmenter

    def load_mask(
        self, image_filename: str, img_shape: tuple[int, int] = None
    ) -> np.ndarray:
        base = os.path.splitext(image_filename)[0]

        # Get image shape for empty mask
        if img_shape is None:
            img_path = os.path.join(self.images_dir, image_filename)
            img = Image.open(img_path)
            img_shape = (img.height, img.width)
            img.close()

        # Load segmentation mask
        mask_path = os.path.join(self.masks_dir, f"{base}.png")
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            return mask
        else:
            # Create empty mask if mask doesn't exist
            return np.zeros(img_shape, dtype=np.uint8)

    def save_augmented_mask(
        self,
        aug_image_filename: str,
        aug_mask: np.ndarray,
    ) -> None:
        base = os.path.splitext(aug_image_filename)[0]

        # Always save mask to maintain consistency
        mask_img = Image.fromarray(aug_mask)
        mask_path = os.path.join(self.masks_dir, f"{base}.png")
        mask_img.save(mask_path)

    def augment_with_mask(
        self, img: np.ndarray, segmentation_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Apply augmentation to both image and mask
        transf = self.augmenter(image=img, mask=segmentation_mask)

        return transf["image"], transf["mask"]

    def augment(self, num_images: int) -> None:
        # Get list of available images
        image_filenames = [
            f
            for f in os.listdir(self.images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for _ in tqdm(range(num_images), desc="Augmenting images and masks"):
            # Get random image
            img_path, img = get_random_img(self.images_dir, image_filenames)
            img_shape = img.shape[:2]  # (height, width)

            # Load corresponding mask
            segmentation_mask = self.load_mask(img_path, img_shape)

            # Augment image and mask together
            aug_img, aug_mask = self.augment_with_mask(img, segmentation_mask)

            # Save augmented image
            aug_img_filename = self.transform_and_save_image(img_path, aug_img)

            # Save augmented mask
            self.save_augmented_mask(aug_img_filename, aug_mask)
