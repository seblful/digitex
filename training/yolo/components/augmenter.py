import os
import random
from PIL import Image

import numpy as np
import cv2

import supervision as sv

import albumentations as A

from tqdm import tqdm

from modules.handlers import LabelHandler

from .converter import Converter
from .utils import get_random_img


class Augmenter:
    def __init__(self,
                 dataset_dir) -> None:
        # Paths
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, 'train')

        self.__transform = None

        self.anns_types = ["polygon", "obb"]

        self.preprocess_funcs = {"polygon": Converter.point_to_polygon,
                                 "obb": Converter.xyxyxyxy_to_polygon}
        self.postprocess_funcs = {"polygon": Converter.polygon_to_point,
                                  "obb": Converter.polygon_to_xyxyxyxy}

        self.img_ext = ".jpg"
        self.anns_ext = ".txt"

        # Handlers
        self.label_handler = LabelHandler()
        self.converter = Converter()

    @property
    def transform(self) -> A.Compose:
        if self.__transform is None:
            self.__transform = A.Compose([
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
                A.SafeRotate(limit=(-3, 3), fill=255, p=0.4)
            ])

        return self.__transform

    def find_name(self, img_name: str) -> str:
        name = os.path.splitext(img_name)[0]

        increment = 1
        while True:
            aug_name = f"{name}_aug_{increment}"
            filename = f"{aug_name}{self.img_ext}"
            filepath = os.path.join(self.train_dir, filename)
            if not os.path.exists(filepath):
                return aug_name
            increment += 1

    def save_anns(self,
                  name: str,
                  points_dict: dict[int, list]) -> None:
        filename = f"{name}{self.anns_ext}"
        filepath = os.path.join(self.train_dir, filename)

        # Write each class and anns to txt
        with open(filepath, 'w') as file:
            if points_dict is None:
                return

            for class_idx, points in points_dict.items():
                for point in points:
                    point = [str(pts) for pts in point]
                    pts = " ".join(point)
                    line = f"{class_idx} {pts}\n"
                    file.write(line)

    def save_image(self,
                   name: str,
                   img: np.ndarray) -> None:
        filename = f"{name}{self.img_ext}"
        filepath = os.path.join(self.train_dir, filename)

        # Save image
        image = Image.fromarray(img)
        image.save(filepath)

    def save(self,
             img_name,
             img: np.ndarray,
             points_dict: dict[int, list]) -> None:

        name = self.find_name(img_name)
        self.save_image(name, img)

        self.save_anns(name, points_dict)

    def create_masks(self,
                     img_name: str,
                     img_width: int,
                     img_height: int,
                     anns_type: str) -> None | dict[int, list]:
        anns_name = os.path.splitext(img_name)[0] + '.txt'
        anns_path = os.path.join(self.train_dir, anns_name)

        points_dict = self.label_handler._read_points(anns_path)

        if not points_dict:
            return None

        masks_dict = {key: [] for key in points_dict.keys()}
        preprocess_func = self.preprocess_funcs[anns_type]

        # Iterate through points, preprocess and convert to mask
        for class_idx, points in points_dict.items():
            for point in points:
                polygon = preprocess_func(point, img_width, img_height)

                # Convert polygon to mask
                mask = sv.polygon_to_mask(polygon, (img_width, img_height))

                masks_dict[class_idx].append(mask)

        return masks_dict

    def create_anns(self,
                    masks_dict: dict[int, list],
                    img_width: int,
                    img_height: int,
                    anns_type: str) -> None | dict[int, list]:
        if masks_dict is None:
            return None

        postprocess_func = self.postprocess_funcs[anns_type]

        points_dict = {key: [] for key in masks_dict.keys()}

        # Iterate through masks and convert to anns
        for class_idx, masks in masks_dict.items():
            for mask in masks:
                polygons = sv.mask_to_polygons(mask)
                polygon = max(polygons, key=cv2.contourArea)
                anns = postprocess_func(
                    polygon, img_width, img_height)
                points_dict[class_idx].append(anns)

        return points_dict

    def augment_image(self,
                      img: np.ndarray,
                      masks_dict: dict[int, list] = None) -> tuple[np.ndarray, None] | tuple[np.ndarray, dict[int, list]]:
        # Case if no masks_dict
        if masks_dict is None:
            transf = self.transform(image=img)
            transf_img = transf['image']

            return (transf_img, None)

        # Obtain masks
        masks = []
        for v in masks_dict.values():
            masks.extend(v)

        # Transform
        transf = self.transform(image=img, masks=masks)
        transf_img = transf['image']
        transf_masks = transf['masks']

        # Create transf_masks_dict
        transf_masks_dict = {key: [] for key in masks_dict.keys()}
        i = 0
        for class_idx, masks in masks_dict.items():
            for _ in range(len(masks)):
                transf_masks_dict[class_idx].append(transf_masks[i])
                i += 1

        return transf_img, transf_masks_dict

    def augment(self,
                anns_type: str,
                num_images: int) -> None:
        assert anns_type in self.anns_types, f"label_type must be one of {self.anns_types}."

        images_listdir = [img_name for img_name in os.listdir(
            self.train_dir) if img_name.endswith(".jpg")]

        for _ in tqdm(range(num_images), desc="Augmenting images"):
            img_name, img = get_random_img(self.train_dir, images_listdir)
            img_height, img_width, _ = img.shape

            masks_dict = self.create_masks(
                img_name, img_width, img_height, anns_type)
            transf_img, transf_masks_dict = self.augment_image(img, masks_dict)
            transf_points_dict = self.create_anns(
                transf_masks_dict, img_width, img_height, anns_type)
            self.save(img_name, transf_img, transf_points_dict)
