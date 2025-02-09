import os
from PIL import Image

import numpy as np
import cv2

import albumentations as A

from tqdm import tqdm

from modules.handlers import ImageHandler, LabelHandler


class Augmenter:
    def __init__(self,
                 dataset_dir) -> None:
        # Paths
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, 'train')

        self.__augmenter = None

        self.label_types = ["polygon", "obb"]

        # Handlers
        self.image_handler = ImageHandler()
        self.label_handler = LabelHandler()

    @property
    def augmenter(self) -> iaa.SomeOf:
        if self.__augmenter is None:
            # Define the augmenter
            aug = iaa.SomeOf((4, None), [
                iaa.Affine(scale=(0.9, 1.05),
                           rotate=(-3, 3),
                           shear=(-2, 3)),
                iaa.PerspectiveTransform(scale=(0.01, 0.05)),
                iaa.CropAndPad(percent=(-0.04, 0.04)),
                iaa.Dropout(p=(0, 0.05)),
                iaa.ImpulseNoise(0.05),
                iaa.GaussianBlur(sigma=(0.0, 1.5)),
                iaa.MultiplyAndAddToBrightness(mul=(0.7, 1.3), add=(-10, 10)),
                iaa.MultiplyHueAndSaturation(mul_hue=(0.9, 1.1)),
                iaa.GammaContrast((0.5, 2.0)),
                iaa.ChangeColorTemperature((1100, 10000))
            ])

            self.__augmenter = aug

        return self.__augmenter

    @staticmethod
    def parse_labels_file(file_path) -> list[dict]:
        '''
        Gets txt file of labels and returns list with dicts, 
        that contains label and polygon points
        '''
        labels = []

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip().split()
                label_class = int(line[0])

                polygon = []

                for i in range(1, len(line), 2):
                    x = float(line[i])
                    y = float(line[i+1])
                    polygon.append((x, y))
                labels.append({'class': label_class, 'polygon': polygon})

        return labels

    @staticmethod
    def load_image(image_path) -> np.ndarray:
        image = Image.open(image_path)
        image.load()
        image_array = np.asarray(image)

        return image_array

    @staticmethod
    def extract_and_convert_polygons(points_with_labels, image_array) -> list[Polygon]:
        '''
        Extracts polygons from points_with_labels dict and convert it 
        from yolo(standartized) format to format with width and height
        and returns original_polygons for one image
        '''
        # Defining image height and image width
        image_height, image_width, _ = image_array.shape
        # Convert points to a Polygon objects
        # Create list for storing polygons for one image
        original_polygons = []
        # Iterating through list of dicts with points and labels
        for polygon_dict in points_with_labels:
            # Convert points to a Keypoint objects
            # Create a list for storing keypoints
            keypoints = []

            # Iterating through points
            for point_x, point_y in polygon_dict['polygon']:

                # Value translation
                translated_point_x = point_x * image_width
                translated_point_y = point_y * image_height

                # Labels of each polygon
                label = polygon_dict['class']

                # Convert points to a Keypoint object
                keypoint = Keypoint(translated_point_x, translated_point_y)

                # Appending keypoint to list with keypoints objects
                keypoints.append(keypoint)

            # Convert keypoints to a Polygon objects
            polygon = Polygon(keypoints, label=label)
            # Appending polygon to list with polygons objects
            original_polygons.append(polygon)

        return original_polygons

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
        self.save_polygons_to_txt(polygons=polygons_aug_i,
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
        points_with_labels = self.parse_labels_file(label_path)

        # Convert original points to a Polygon objects and convert points
        original_polygons = self.extract_and_convert_polygons(
            points_with_labels, image_array)

        # Augment images and polygons
        for num_aug in range(aug_factor):
            images_aug_i, polygons_aug_i = augmenter(
                image=image_array, polygons=original_polygons)

            self.save_augmented_images_with_labels(image_array=image_array,
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
            self.save_augmented_zero_images(image_name=image_name,
                                            images_aug_i=images_aug_i,
                                            num_aug=num_aug,
                                            train_dataset_dir=train_dataset_dir)

    def augment(self,
                label_type: str,
                num_images: int) -> None:
        assert label_type in self.label_types, f"label_type must be one of {self.label_types}."

        images_listdir = [i for i in os.listdir(
            self.train_dir) if i.endswith(".jpg")]

        for _ in tqdm(range(num_images), desc="Augmenting images"):
            rand_image, rand_image_name = self.image_handler.get_random_image(
                images_listdir, self.train_dir)
            rand_label_name, _ = self.label_handler.get_random_label(
                rand_image_name, self.train_dir)

            if rand_label_name is None:
                # zero images
                pass

            else:
                # augment non zero images
                pass
