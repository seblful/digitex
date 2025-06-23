import os
from PIL import Image

import numpy as np
import albumentations as A
from tqdm import tqdm

from digitex.core.utils import get_random_img
from digitex.core.processors.file import FileProcessor

from .annotation import (
    RelativeKeypointsObject,
    AbsoluteKeypointsObject,
)

from .data import MaskGenerator
from .utils import create_abs_kps_obj_from_label


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

    def save_image(self, img_path: str, img: np.ndarray) -> None:
        aug_img_filename = self.find_path(img_path)
        aug_img_path = os.path.join(self.images_dir, aug_img_filename)

        # Save image
        image = Image.fromarray(img)
        image.save(aug_img_path)

        return aug_img_filename


class MaskAugmenter(BaseAugmenter):
    def __init__(
        self, raw_dir: str, dataset_dir: str, mask_radius_ratio: float = 0.02
    ) -> None:
        super().__init__(raw_dir, dataset_dir)
        self.masks_dir = os.path.join(self.train_dir, "masks")

        # Load original annotations for keypoint-based augmentation
        self.annotations_json_path = os.path.join(raw_dir, "anns.json")
        self.annotations_dict = FileProcessor.read_json(
            json_path=self.annotations_json_path
        )

        # Initialize mask generator for creating masks from augmented keypoints
        self.mask_generator = MaskGenerator(mask_radius_ratio=mask_radius_ratio)

    @property
    def augmenter(self) -> A.Compose:
        if self._augmenter is None:
            # Create transforms that support keypoints
            keypoint_transforms = [
                # Geometric transforms that affect keypoints
                A.Affine(
                    scale=[0.92, 1.08],
                    translate_percent=[-0.05, 0.05],
                    rotate=[-3, 3],
                    fill=255,
                    p=0.4,
                ),
                A.SafeRotate(limit=(-5, 5), fill=255, p=0.4),
                A.RandomScale(scale_limit=0.1, p=0.4),
                A.Pad(padding=[15, 15], fill=255, p=0.4),
                # Photometric transforms (don't affect keypoints)
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
                A.CoarseDropout(fill=255, p=0.1),
            ]

            # Create compose with keypoint support and preserve invisible keypoints
            self._augmenter = A.Compose(
                keypoint_transforms,
                keypoint_params=A.KeypointParams(
                    format="xy",
                    remove_invisible=False,  # Keep invisible keypoints!
                ),
            )
        return self._augmenter

    def _get_image_dims(self, image_filename: str) -> tuple[int, int]:
        image_path = os.path.join(self.images_dir, image_filename)
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        return image_width, image_height

    def _save_augmented_mask(self, aug_image_filename: str, mask: np.ndarray) -> None:
        base = os.path.splitext(aug_image_filename)[0]
        mask_path = os.path.join(self.masks_dir, f"{base}.png")

        mask_img = Image.fromarray(mask, mode="L")
        mask_img.save(mask_path)

    def _augment_img(
        self, img: np.ndarray, kps_obj: RelativeKeypointsObject
    ) -> tuple[np.ndarray, list]:
        # Transform without keypoints
        if not kps_obj.keypoints:
            transf = self.augmenter(image=img)
            transf_img = transf["image"]

            return (transf_img, [])

        # Retrieve all visible coordinates
        vis_coords = kps_obj.get_vis_coords()

        # Transform with keypoints
        transf = self.augmenter(image=img, keypoints=vis_coords)
        transf_img = transf["image"]
        transf_label = [
            [int(value[0]), int(value[1]), 1] for value in transf["keypoints"]
        ]

        return transf_img, transf_label

    def _generate_mask(
        self, kps_obj: AbsoluteKeypointsObject, img_width: int, img_height: int
    ) -> np.ndarray:
        label = kps_obj.get_label()
        mask = self.mask_generator.generate_mask_from_label(
            label, img_width, img_height
        )

        return mask

    def augment(self, num_images: int) -> None:
        # Get list of available images
        image_filenames = [
            f
            for f in os.listdir(self.images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for _ in tqdm(range(num_images), desc="Augmenting images"):
            # Get random image
            img_path, img = get_random_img(self.images_dir, image_filenames)
            orig_height, orig_width = img.shape[:2]

            # Create KeypointsObject from labels
            abs_kps_obj = create_abs_kps_obj_from_label(
                label=self.annotations_dict[img_path],
                clip=False,
                img_width=orig_width,
                img_height=orig_height,
            )

            # Augment
            transf_img, transf_label = self._augment_img(img, abs_kps_obj)
            transf_height, transf_width = transf_img.shape[:2]

            # Create transformed keypoints object from transf_label
            transf_abs_kps_obj = create_abs_kps_obj_from_label(
                label=transf_label,
                clip=True,
                img_width=transf_width,
                img_height=transf_height,
                num_keypoints=len(abs_kps_obj.keypoints),
            )
            transf_mask = self._generate_mask(
                transf_abs_kps_obj, transf_width, transf_height
            )

            # Save augmented image
            aug_img_path = self.save_image(img_path, transf_img)

            # Save generated mask
            self._save_augmented_mask(aug_img_path, transf_mask)
