import os
from PIL import Image

import numpy as np
import cv2

from numpy._typing._array_like import NDArray
from tqdm import tqdm

import imgaug.augmenters as iaa
from imgaug.augmenters import SomeOf
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class Augmenter:
    def __init__(self,
                 dataset_path,
                 zero_augmenter_factor,
                 labels_augmenter_factor):

        self.dataset_path = dataset_path
        self.__train_dataset_path = None

        self.__images_labels_dict = None

        self.__augmenter = None
        self.zero_augmenter_factor = zero_augmenter_factor
        self.labels_augmenter_factor = labels_augmenter_factor

    @property
    def augmenter(self) -> SomeOf:
        if self.__augmenter is None:
            # Define the augmenter
            aug = iaa.SomeOf((2, 7), [
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-10, 10),
                           scale={"x": (0.7, 1.1), "y": (0.7, 1.1)}),
                iaa.PiecewiseAffine(scale=(0.01, 0.03)),  # takes much time
                iaa.Dropout(p=(0, 0.05)),
                iaa.ImpulseNoise(0.05),
                iaa.AverageBlur(k=((4, 7), (1, 3))),
                iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-20, 20)),
                iaa.GammaContrast((0.5, 2.0), per_channel=True),
                iaa.Grayscale(alpha=(0.0, 0.5)),
                iaa.MultiplyHueAndSaturation(mul_hue=(0.7, 1.7)),
                iaa.ChangeColorTemperature((1100, 15000)),
                iaa.RemoveSaturation(1.0)
            ])

            # aug = iaa.Sequential([iaa.ImpulseNoise(0.05)])

            self.__augmenter = aug

        return self.__augmenter

    @property
    def train_dataset_path(self) -> str:
        if self.__train_dataset_path is None:
            self.__train_dataset_path = os.path.join(
                self.dataset_path, 'train')

        return self.__train_dataset_path

    @property
    def images_labels_dict(self) -> str:
        '''
        Dict with names of images with corresponding 
        names of label
        '''
        if self.__images_labels_dict is None:
            self.__images_labels_dict = self.__create_images_labels_dict()

        return self.__images_labels_dict

    @staticmethod
    def is_file_empty(file_path) -> bool:
        """
        Check if a text file is empty.
        Returns True if the file is empty, False otherwise.
        """
        is_file = os.path.isfile(file_path) and os.path.getsize(file_path) == 0

        return is_file

    def __create_images_labels_dict(self) -> dict[str, str]:
        # List of all images and labels in directory
        images = [image for image in os.listdir(
            self.train_dataset_path) if image.endswith(('.jpg', '.png'))]
        labels = [label for label in os.listdir(
            self.train_dataset_path) if label.endswith('.txt')]

        # Create a dictionary to store the images and labels names
        images_labels = {}
        for image in images:
            label = image.rstrip('.jpg') + '.txt'
            label_path = os.path.join(self.train_dataset_path, label)

            if label in labels and Augmenter.is_file_empty(label_path) is not True:
                images_labels[image] = label
            else:
                images_labels[image] = None

        return images_labels

    @staticmethod
    def parse_labels_file(label_path) -> list[dict]:
        std_bboxes = []

        with open(label_path, 'r') as lab:
            for line in lab:
                label_with_bbox = [float(i) for i in line.split()]
                label = int(label_with_bbox[0])
                bbox = label_with_bbox[1:]

                std_bboxes.append({'class': label, 'bbox': bbox})

        return std_bboxes

    @staticmethod
    def convert_yolo_to_imgaug_bbox(yolo_bboxes,
                                    image_array) -> list[BoundingBox]:
        # Retrieve image_height, image_width
        try:
            image_height, image_width, _ = image_array.shape
        except:
            image_height, image_width = image_array.shape

        imgaug_bboxes = []

        for bbox in yolo_bboxes:
            # Extract label and bbox points
            label = bbox['class']
            x_center, y_center, width, height = bbox['bbox']

            # Convert bbox points
            x1 = (x_center - width / 2) * image_width
            y1 = (y_center - height / 2) * image_height
            x2 = (x_center + width / 2) * image_width
            y2 = (y_center + height / 2) * image_height

            imgaug_bboxes.append(BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2, label=label))

        return imgaug_bboxes

    def convert_imgaug_to_yolo_bbox(image_array,
                                    imgaug_bboxes) -> list[dict]:
        # Extract image height and image width
        image_height, image_width, _ = image_array.shape

        std_bboxes = []

        for bbox in imgaug_bboxes.bounding_boxes:
            # Extract points and label
            x1, y1, x2, y2, label = bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.label

            # Convert points
            x_center = (x1 + x2) / 2 / image_width
            y_center = (y1 + y2) / 2 / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            yolo_bbox = [x_center, y_center, width, height]
            # yolo_bbox = [float(i) for i in yolo_bbox]

            std_bboxes.append({'class': label, 'bbox': yolo_bbox})

        return std_bboxes

    @staticmethod
    def load_image(image_path) -> np.ndarray:
        image = Image.open(image_path)
        image.load()
        image_array = np.asarray(image)

        return image_array

    def save_bboxes_to_txt(image_array,
                           bboxes_aug_i,
                           label_save_path) -> None:
        '''
        Takes augmented boxes and saves them in save path as txt file
        '''
        bboxes_dicts = Augmenter.convert_imgaug_to_yolo_bbox(image_array=image_array,
                                                             imgaug_bboxes=bboxes_aug_i)

        with open(label_save_path, 'w') as file:
            # Iterating through each box and write coordinates
            for bbox_dict in bboxes_dicts:
                item_str = f"{bbox_dict['class']} {
                    ' '.join(map(str, bbox_dict['bbox']))}"
                file.write(item_str + '\n')

    @staticmethod
    def save_augmented_images_with_labels(image_array,
                                          image_name,
                                          images_aug_i,
                                          bboxes_aug_i,
                                          number_of_augmentation,
                                          train_dataset_path) -> None:

        # Defining names of images and labels
        new_name_of_file = f"{image_name.rstrip('.jpg')}_aug_{
            str(number_of_augmentation + 1)}"
        new_name_of_image, new_name_of_label = [
            new_name_of_file + file_format for file_format in ('.jpg', '.txt')]

        # Defining path to save images and labels
        image_save_path = os.path.join(
            train_dataset_path, new_name_of_image)
        label_save_path = os.path.join(
            train_dataset_path, new_name_of_label)

        # Formatting and save augmented image
        rgb_image = cv2.cvtColor(images_aug_i, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_save_path, rgb_image)
        # Save augmented bboxes to txt file in yolov8 format
        Augmenter.save_bboxes_to_txt(image_array=image_array,
                                     bboxes_aug_i=bboxes_aug_i,
                                     label_save_path=label_save_path)

    @staticmethod
    def save_augmented_zero_images(image_name,
                                   images_aug_i,
                                   number_of_augmentation,
                                   train_dataset_path) -> None:
        # Defining path to save images
        new_name_of_file = f"{image_name.rstrip('.jpg')}_aug_{
            str(number_of_augmentation + 1)}.jpg"
        image_save_path = os.path.join(train_dataset_path, new_name_of_file)

        # Formatting and save augmented image
        rgb_image = cv2.cvtColor(images_aug_i, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_save_path, rgb_image)

    @staticmethod
    def augment_images_with_labels(train_dataset_path,
                                   image_array,
                                   image_name,
                                   label_name,
                                   augmenter,
                                   augmenter_factor) -> None:
        label_path = os.path.join(train_dataset_path, label_name)

        # Create list of dicts with labels for one image and transform it
        yolo_bboxes = Augmenter.parse_labels_file(label_path)
        imgaug_bboxes = Augmenter.convert_yolo_to_imgaug_bbox(
            yolo_bboxes, image_array)
        bboxes_on_image = BoundingBoxesOnImage(
            imgaug_bboxes, shape=image_array.shape)

        # Augment images and bboxes
        for number_of_augmentation in range(augmenter_factor):
            images_aug_i, bboxes_aug_i = augmenter(
                image=image_array, bounding_boxes=bboxes_on_image)

            Augmenter.save_augmented_images_with_labels(image_array=image_array,
                                                        image_name=image_name,
                                                        images_aug_i=images_aug_i,
                                                        bboxes_aug_i=bboxes_aug_i,
                                                        number_of_augmentation=number_of_augmentation,
                                                        train_dataset_path=train_dataset_path)

    @staticmethod
    def augment_zero_images(train_dataset_path,
                            image_array,
                            image_name,
                            augmenter,
                            augmenter_factor) -> None:

        if augmenter_factor == 0:
            return

        # Augment images
        for number_of_augmentation in range(augmenter_factor):
            try:
                images_aug_i = augmenter(image=image_array)
                # Save images
                Augmenter.save_augmented_zero_images(image_name=image_name,
                                                     images_aug_i=images_aug_i,
                                                     number_of_augmentation=number_of_augmentation,
                                                     train_dataset_path=train_dataset_path)
            except AssertionError as wrong_n_chann_error:
                continue

    def augment(self) -> None:
        # Return if augmenter factor is 0
        if self.zero_augmenter_factor == 0 and self.labels_augmenter_factor == 0:
            return
        # Iterating through each image and label from raw directory
        for image_name, label_name in tqdm(self.images_labels_dict.items(), desc='Augmenting Images'):
            # Load image
            image_path = os.path.join(self.train_dataset_path, image_name)
            image_array = Augmenter.load_image(image_path=image_path)

            if label_name is not None:
                Augmenter.augment_images_with_labels(train_dataset_path=self.train_dataset_path,
                                                     image_array=image_array,
                                                     image_name=image_name,
                                                     label_name=label_name,
                                                     augmenter=self.augmenter,
                                                     augmenter_factor=self.labels_augmenter_factor)
            else:
                Augmenter.augment_zero_images(train_dataset_path=self.train_dataset_path,
                                              image_array=image_array,
                                              image_name=image_name,
                                              augmenter=self.augmenter,
                                              augmenter_factor=self.zero_augmenter_factor)
