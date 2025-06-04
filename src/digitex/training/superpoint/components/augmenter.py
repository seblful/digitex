import os
from PIL import Image

import numpy as np
import albumentations as A
from tqdm import tqdm

from digitex.core.processors.file import FileProcessor
from digitex.core.utils import get_random_img
from digitex.training.superpoint.components.annotation import (
    RelativeKeypoint,
    AbsoluteKeypoint,
    RelativeKeypointsObject,
    AbsoluteKeypointsObject,
)


class BaseAugmenter:
    def __init__(self, raw_dir: str, dataset_dir: str) -> None:
        # Paths
        self.raw_dir = raw_dir
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, "train")

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
        aug_img_path = os.path.join(self.train_dir, aug_img_filename)

        # Save image
        image = Image.fromarray(img)
        image.save(aug_img_path)

        return aug_img_filename
