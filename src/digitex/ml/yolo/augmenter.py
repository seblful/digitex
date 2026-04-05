import logging
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import albumentations as A
import supervision as sv
from tqdm import tqdm

from digitex.core.handlers import LabelHandler

from .converter import Converter
from .utils import get_random_img

logger = logging.getLogger(__name__)


class Augmenter:
    def __init__(self, classes: dict[int, str], dataset_dir: str | Path) -> None:
        self.classes = classes
        self.dataset_dir = Path(dataset_dir)
        self.train_dir = self.dataset_dir / "train"

        self.img_ext = ".jpg"
        self.anns_ext = ".txt"

        self._transforms = None
        self._augmenter = None

    @property
    def transforms(self) -> list:
        if self._transforms is None:
            self._transforms = [
                A.AdditiveNoise(p=0.3),
                A.Downscale(scale_range=(0.4, 0.9), p=0.3),
                A.RGBShift(p=0.3),
                A.RingingOvershoot(p=0.3),
                A.Spatter(mean=(0.5, 0.6), p=0.2),
                A.ToGray(p=0.4),
                A.ChannelShuffle(p=0.3),
                A.Emboss(p=0.3),
                A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
                A.HueSaturationValue(p=0.3),
                A.MedianBlur(p=0.3),
                A.PlanckianJitter(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomShadow(shadow_intensity_range=(0.1, 0.4), p=0.3),
                A.SaltAndPepper(amount=(0.01, 0.03), p=0.2),
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
                A.Affine(scale=(0.92, 1.08), fill=255, p=0.4),
                A.CoarseDropout(fill=255, p=0.1),
                A.Pad(padding=(15, 15), fill=255, p=0.4),
                A.RandomScale(p=0.4),
                A.SafeRotate(limit=(-3, 3), fill=255, p=0.4),
            ]

        return self._transforms

    @property
    def augmenter(self) -> A.Compose | None:
        if self._augmenter is None:
            self._augmenter = A.Compose(self.transforms)
        return self._augmenter

    @property
    def id2label(self) -> dict[int, str]:
        return self.classes

    @property
    def label2id(self) -> dict[str, int]:
        return {v: k for k, v in self.classes.items()}

    def find_name(self, img_name: str) -> str:
        name = Path(img_name).stem

        increment = 1
        while True:
            aug_name = f"{name}_aug_{increment}"
            filename = f"{aug_name}{self.img_ext}"
            filepath = Path(self.train_dir) / filename
            if not filepath.exists():
                return aug_name
            increment += 1

    def save_anns(self, name: str, points_dict: dict[int, list] | None = None) -> None:
        pass

    def save_image(self, name: str, img: np.ndarray) -> None:
        filename = f"{name}{self.img_ext}"
        filepath = self.train_dir / filename

        image = Image.fromarray(img)
        image.save(str(filepath))

    def save(
        self, img_name, img: np.ndarray, points_dict: dict[int, list] | None = None
    ) -> None:

        name = self.find_name(img_name)
        self.save_image(name, img)
        self.save_anns(name, points_dict)


class PolygonAugmenter(Augmenter):
    def __init__(self, classes: dict[int, str], dataset_dir: str) -> None:
        super().__init__(classes, dataset_dir)

        self.preprocess_func = Converter.point_to_polygon
        self.postprocess_func = Converter.polygon_to_point

    def save_anns(self, name: str, points_dict: dict[int, list] | None = None) -> None:
        filename = f"{name}{self.anns_ext}"
        filepath = self.train_dir / filename

        with open(str(filepath), "w") as file:
            if points_dict is None:
                return

            for class_idx, points in points_dict.items():
                for point in points:
                    point = [str(pts) for pts in point]
                    pts = " ".join(point)
                    line = f"{class_idx} {pts}\n"
                    file.write(line)

    def create_masks(
        self, img_name: str, img_width: int, img_height: int
    ) -> dict[int, list] | None:
        anns_name = Path(img_name).stem + ".txt"
        anns_path = self.train_dir / anns_name

        points_dict = LabelHandler._read_points(str(anns_path))

        if not points_dict:
            return None

        masks_dict = {key: [] for key in points_dict.keys()}

        for class_idx, points in points_dict.items():
            for point in points:
                polygon = self.preprocess_func(point, img_width, img_height)

                mask = sv.polygon_to_mask(polygon, (img_width, img_height))

                masks_dict[class_idx].append(mask)

        return masks_dict

    def create_anns(
        self, masks_dict: dict[int, list] | None, img_width: int, img_height: int
    ) -> None | dict[int, list]:
        if masks_dict is None:
            return None

        points_dict = {key: [] for key in masks_dict.keys()}

        for class_idx, masks in masks_dict.items():
            for mask in masks:
                polygons = sv.mask_to_polygons(mask)
                polygon = max(polygons, key=cv2.contourArea)  # ty: ignore[no-matching-overload]
                anns = self.postprocess_func(polygon, img_width, img_height)
                points_dict[class_idx].append(anns)

        return points_dict

    def augment_img(
        self, img: np.ndarray, masks_dict: dict[int, list] | None = None
    ) -> tuple[np.ndarray, None] | tuple[np.ndarray, dict[int, list]]:

        if masks_dict is None:
            aug = self.augmenter
            if aug is None:
                raise RuntimeError("Failed to create augmenter")
            transf = aug(image=img)
            transf_img = transf["image"]

            return (transf_img, None)

        masks = []
        for v in masks_dict.values():
            masks.extend(v)

        aug = self.augmenter
        if aug is None:
            raise RuntimeError("Failed to create augmenter")
        transf = aug(image=img, masks=masks)
        transf_img = transf["image"]
        transf_masks = transf["masks"]

        transf_masks_dict = {key: [] for key in masks_dict.keys()}
        i = 0
        for class_idx, masks in masks_dict.items():
            for _ in range(len(masks)):
                transf_masks_dict[class_idx].append(transf_masks[i])
                i += 1

        return transf_img, transf_masks_dict

    def augment(self, num_images: int) -> None:
        images_listdir = [
            img_name
            for img_name in os.listdir(self.train_dir)
            if img_name.endswith(".jpg")
        ]

        for _ in tqdm(range(num_images), desc="Augmenting images"):
            img_name, img = get_random_img(self.train_dir, images_listdir)

            orig_height, orig_width = img.shape[:2]
            masks_dict = self.create_masks(img_name, orig_width, orig_height)

            transf_img, transf_masks_dict = self.augment_img(img, masks_dict)
            transf_height, transf_width = transf_img.shape[:2]

            transf_points_dict = self.create_anns(
                transf_masks_dict, transf_width, transf_height
            )
            self.save(img_name, transf_img, transf_points_dict)
