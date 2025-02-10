import os
import random
from PIL import Image

import numpy as np
import cv2

import supervision as sv

import albumentations as A

from tqdm import tqdm

from modules.handlers import ImageHandler, LabelHandler


class Polygon:
    pass


class Keypoint:
    pass


class Augmenter:
    def __init__(self,
                 dataset_dir) -> None:
        # Paths
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, 'train')

        self.__transform = None

        self.label_types = ["polygon", "obb"]

        self.prepare_funcs = {"polygon": self.point_to_polygon,
                              "obb": self.obb_to_polygon}

        # Handlers
        self.image_handler = ImageHandler()
        self.label_handler = LabelHandler()

    @property
    def transform(self) -> A.Compose:
        if self.__transform is None:
            self.__transform = A.Compose([
                A.Affine(scale=0.9, p=0.7),
                A.Perspective(scale=(0.01, 0.05), p=0.5),
                A.CropAndPad(percent=(-0.04, 0.04), p=0.5),
                A.CoarseDropout(p=0.2),
                A.ISONoise(color_shift=(0.01, 0.05), p=0.2),
                A.GaussianBlur(blur_limit=(0, 3), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3),
                                           contrast_limit=(-0.3, 0.3),
                                           p=0.5),
                A.HueSaturationValue(hue_shift_limit=10,
                                     sat_shift_limit=20,
                                     val_shift_limit=10,
                                     p=0.5),
                A.RandomGamma(gamma_limit=(50, 200), p=0.5)
            ])

        return self.__transform

    @staticmethod
    def save_polygons_to_txt(polygons, image_width, image_height, filepath) -> None:
        '''
        Saves list of polygons to txt file in yolov8 format
        '''
        with open(filepath, 'w') as f:
            # Iterating through each polygon
            for polygon in polygons:
                # Write label of 1 polygon
                f.write(f"{polygon.label} ")

                # Iterating through each point
                for point in polygon.exterior:
                    x, y = point[0], point[1]
                    x = x / image_width
                    y = y / image_height
                    # Check if label is not out of coordinates
                    if (x < 0 or x > 1) or (y < 0 or y > 1):
                        continue
                    # Write each coordinate
                    f.write(f"{x} {y} ")
                f.write('\n')

        return None

    @staticmethod
    def save_augmented_images_with_labels(image_array,
                                          image_name,
                                          images_aug_i,
                                          polygons_aug_i,
                                          num_aug,
                                          train_dataset_dir) -> None:

        # Defining image height and image width
        image_height, image_width, _ = image_array.shape

        # Defining names of images and labels
        new_file_name = f"{image_name.rstrip('.jpg')}_aug_{
            str(num_aug + 1)}"
        new_image_name, new_label_name = [
            new_file_name + file_format for file_format in ('.jpg', '.txt')]

        # Defining path to save images and labels
        image_save_path = os.path.join(
            train_dataset_dir, new_image_name)
        label_save_path = os.path.join(
            train_dataset_dir, new_label_name)

        # Save augmented image
        Image.fromarray(images_aug_i).save(image_save_path)

        # Save augmented polygons to txt file in yolov8 format
        Augmenter.save_polygons_to_txt(polygons=polygons_aug_i,
                                       image_width=image_width,
                                       image_height=image_height,
                                       filepath=label_save_path)

    @staticmethod
    def save_augmented_zero_images(image_name,
                                   images_aug_i,
                                   num_aug,
                                   train_dataset_dir) -> None:
        # Save images
        new_file_name = f"{image_name.rstrip('.jpg')}_aug_{
            str(num_aug + 1)}.jpg"
        image_save_path = os.path.join(train_dataset_dir, new_file_name)
        cv2.imwrite(image_save_path, images_aug_i)

    @staticmethod
    def augment_images_with_labels(train_dataset_dir,
                                   image_array,
                                   image_name,
                                   label_name,
                                   augmenter,
                                   aug_factor) -> None:
        label_path = os.path.join(train_dataset_dir, label_name)
        # Create list of dicts with labels for one image
        points_with_labels = Augmenter.parse_labels_file(label_path)

        # Convert original points to a Polygon objects and convert points
        original_polygons = Augmenter.extract_and_convert_polygons(
            points_with_labels, image_array)

        # Augment images and polygons
        for num_aug in range(aug_factor):
            images_aug_i, polygons_aug_i = augmenter(
                image=image_array, polygons=original_polygons)

            Augmenter.save_augmented_images_with_labels(image_array=image_array,
                                                        image_name=image_name,
                                                        images_aug_i=images_aug_i,
                                                        polygons_aug_i=polygons_aug_i,
                                                        num_aug=num_aug,
                                                        train_dataset_dir=train_dataset_dir)

    @staticmethod
    def augment_zero_images(train_dataset_dir,
                            image_array,
                            image_name,
                            augmenter,
                            aug_factor) -> None:

        # Augment images
        for num_aug in range(aug_factor):
            images_aug_i = augmenter(image=image_array)
            # Save images
            Augmenter.save_augmented_zero_images(image_name=image_name,
                                                 images_aug_i=images_aug_i,
                                                 num_aug=num_aug,
                                                 train_dataset_dir=train_dataset_dir)

    def get_random_img(self,
                       images_listdir: list[str]) -> tuple[np.ndarray, str]:
        img_name = random.choice(images_listdir)
        img_path = os.path.join(self.train_dir, img_name)
        image = Image.open(img_path)
        img = np.array(image)

        return img_name, img

    def obb_to_polygon(self,
                       obb: list[float],
                       img_width: int,
                       img_height: int) -> np.ndarray:
        obb = [obb[i] * (img_width if i % 2 == 0 else img_height)
               for i in range(len(obb))]
        obb = np.array(obb)
        polygon = sv.xyxy_to_polygons(obb)

        return polygon

    def point_to_polygon(self,
                         point: list[float],
                         img_width: int,
                         img_height: int) -> np.ndarray:
        polygon = list(zip(point[::2], point[1::2]))
        polygon = [(int(x * img_width), int(y * img_height))
                   for x, y in polygon]
        polygon = np.array(polygon)

        return polygon

    def create_masks(self,
                     img_name: str,
                     img: np.ndarray,
                     anns_type: str) -> None | dict[int, list]:
        anns_name = os.path.splitext(img_name)[0] + '.txt'
        anns_path = os.path.join(self.train_dir, anns_name)

        if not os.path.exists(anns_path):
            return None

        prepare_func = self.prepare_funcs[anns_type]

        points_dict = self.label_handler._read_points(anns_path)
        masks_dict = {key: [] for key in points_dict.keys()}

        img_height, img_width, _ = img.shape

        # Iterate through points, preprocess and convert to mask
        for class_idx, points in points_dict.items():
            for point in points:
                polygon = prepare_func(point, img_width, img_height)

                # Convert polygon to mask
                mask = sv.polygon_to_mask(polygon, (img_width, img_height))

                masks_dict[class_idx].append(mask)

        return masks_dict

    def augment_image(self,
                      img: np.ndarray,
                      masks_dict: dict[int, list] = None) -> tuple[np.ndarray, None] | tuple[np.ndarray, dict[int, list]]:
        # Obtain masks
        masks = []
        for v in masks_dict.values():
            masks.extend(v)

        masks = []

        # Transform
        if not masks:
            transf = self.transform(image=img)
            transf_img = transf['image']

            return (transf_img, None)

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
        assert anns_type in self.label_types, f"label_type must be one of {self.label_types}."

        images_listdir = [i for i in os.listdir(
            self.train_dir) if i.endswith(".jpg")]

        for _ in tqdm(range(num_images), desc="Augmenting images"):
            img_name, img = self.get_random_img(images_listdir)
            masks_dict = self.create_masks(img_name, img, anns_type)
            transf_img, transf_masks_dict = self.augment_image(img, masks_dict)
