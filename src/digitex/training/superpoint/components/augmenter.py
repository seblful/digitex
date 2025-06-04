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


class KeypointAugmenter(BaseAugmenter):
    def __init__(self, raw_dir: str, dataset_dir: str) -> None:
        super().__init__(raw_dir, dataset_dir)

    @property
    def augmenter(self) -> A.Compose:
        if self._augmenter is None:
            augmenter = A.Compose(
                self.transforms,
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )

        return augmenter

    def create_rel_kps_from_label(
        self, label: list[list], clip: bool
    ) -> list[RelativeKeypoint]:
        kps = []

        for value in label:
            kp = RelativeKeypoint(value[0], value[1], int(value[2]))
            if clip:
                kp.clip()
            kps.append(kp)

        return kps

    def create_abs_kps_from_label(
        self,
        label: list[list],
        clip: bool,
        img_width: int = None,
        img_height: int = None,
    ) -> list[AbsoluteKeypoint]:
        kps = []

        for value in label:
            kp = AbsoluteKeypoint(value[0], value[1], int(value[2]))
            if clip:
                kp.clip(img_width, img_height)
            kps.append(kp)

        return kps

    def create_rel_kps_obj_from_label(
        self, label: list[list], clip: bool
    ) -> RelativeKeypointsObject:
        if not label:
            return RelativeKeypointsObject(0, [], 0)

        kps = self.create_rel_kps_from_label(label, clip)
        kps_obj = RelativeKeypointsObject(
            0,
            kps,
            len(kps),
        )

        return kps_obj

    def create_abs_kps_obj_from_label(
        self,
        label: list[list],
        clip: bool,
        img_width: int = None,
        img_height: int = None,
        num_keypoints: int = None,
    ) -> AbsoluteKeypointsObject:
        if not label:
            return AbsoluteKeypointsObject(0, [], 0)

        kps = self.create_abs_kps_from_label(label, clip, img_width, img_height)
        num_keypoints = num_keypoints if num_keypoints is not None else len(kps)
        kps_obj = AbsoluteKeypointsObject(0, kps, num_keypoints)

        return kps_obj

    def augment_img(
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

    def augment(self, num_images) -> None:
        label_path = os.path.join(self.train_dir, "labels.json")
        labels_dict = FileProcessor.read_json(label_path)

        for _ in tqdm(range(num_images), desc="Augmenting images"):
            # Get random img
            img_path, img = get_random_img(self.train_dir, list(labels_dict.keys()))
            orig_height, orig_width = img.shape[:2]

            # Create KeypointsObject from labels
            rel_kps_obj = self.create_rel_kps_obj_from_label(
                labels_dict[img_path], clip=False
            )
            abs_kps_obj = rel_kps_obj.to_absolute(orig_width, orig_height, clip=False)

            # Augment
            transf_img, transf_label = self.augment_img(img, abs_kps_obj)
            transf_height, transf_width = transf_img.shape[:2]

            # Create transformed keypoints object from transf_label
            transf_abs_kps_obj = self.create_abs_kps_obj_from_label(
                label=transf_label,
                clip=True,
                img_width=transf_width,
                img_height=transf_height,
                num_keypoints=len(abs_kps_obj.keypoints),
            )
            transf_rel_kps_obj = transf_abs_kps_obj.to_relative(
                transf_width, transf_height, clip=False
            )

            # Save augmented image
            aug_img_path = self.save_image(img_path, transf_img)

            # Add label to labels_dict
            labels_dict[aug_img_path] = transf_rel_kps_obj.get_label()

        # Save annotation
        FileProcessor.write_json(labels_dict, label_path)

        return None
