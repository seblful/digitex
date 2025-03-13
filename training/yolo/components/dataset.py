from typing import LiteralString, List, Dict

import os
import shutil
import random

from .annotation import AnnotationCreator


class DatasetCreator:
    def __init__(self,
                 raw_dir,
                 dataset_dir,
                 train_split=0.8) -> None:

        self.raw_dir = raw_dir
        self.dataset_dir = dataset_dir

        self.__images_path = None  # Raw images
        self.__labels_path = None  # Raw labels

        self.__train_dir = None
        self.__val_dir = None
        self.__test_dir = None

        self.__images_labels_dict = None
        self.__classes_path = None
        self.__data_yaml_path = None

        self.train_split = train_split
        self.val_split = 0.6 * (1 - self.train_split)
        self.test_split = 1 - self.train_split - self.val_split

        self.anns_types = ["polygon", "obb", "keypoint"]

        self.anns_creator = AnnotationCreator(raw_dir=raw_dir)

    @property
    def images_path(self) -> LiteralString | str:
        if self.__images_path is None:
            self.__images_path = os.path.join(
                self.raw_dir, 'images')

        return self.__images_path

    @property
    def labels_path(self) -> LiteralString | str:
        if self.__labels_path is None:
            self.__labels_path = os.path.join(
                self.raw_dir, 'labels')

        return self.__labels_path

    @property
    def train_dir(self) -> LiteralString | str:
        if self.__train_dir is None:
            train_dir = os.path.join(self.dataset_dir, 'train')
            # Creating folder for train set
            os.makedirs(train_dir, exist_ok=True)
            self.__train_dir = train_dir

        return self.__train_dir

    @property
    def val_dir(self) -> LiteralString | str:
        if self.__val_dir is None:
            val_dir = os.path.join(self.dataset_dir, 'val')
            # Creating folder for val set
            os.makedirs(val_dir, exist_ok=True)
            self.__val_dir = val_dir

        return self.__val_dir

    @property
    def test_dir(self) -> LiteralString | str:
        if self.__test_dir is None:
            test_dir = os.path.join(self.dataset_dir, 'test')
            # Creating folder for test set
            os.makedirs(test_dir, exist_ok=True)
            self.__test_dir = test_dir

        return self.__test_dir

    @property
    def data_yaml_path(self) -> LiteralString | str:
        if self.__data_yaml_path is None:
            self.__data_yaml_path = os.path.join(
                self.dataset_dir, 'data.yaml')

        return self.__data_yaml_path

    @property
    def classes_path(self) -> LiteralString | str:
        if self.__classes_path is None:
            self.__classes_path = os.path.join(
                self.raw_dir, 'classes.txt')

        return self.__classes_path

    @property
    def images_labels_dict(self) -> Dict[str, str]:
        '''
        Dict with names of images with corresponding 
        names of label
        '''
        if self.__images_labels_dict is None:
            self.__images_labels_dict = self.__create_images_labels_dict()

        return self.__images_labels_dict

    def __create_images_labels_dict(self, shuffle=True) -> Dict[str, str]:
        # List of all images and labels in directory
        images = os.listdir(self.images_path)
        labels = os.listdir(self.labels_path)

        # Create a dictionary to store the images and labels names
        images_labels = {}
        for image in images:
            label = image.rstrip('.jpg') + '.txt'

            if label in labels:
                images_labels[image] = label
            else:
                images_labels[image] = None

        if shuffle:
            # Shuffle the data
            keys = list(images_labels.keys())
            random.shuffle(keys)
            images_labels = {key: images_labels[key] for key in keys}

        return images_labels

    @staticmethod
    def read_classes_file(classes_path) -> List[str]:
        with open(classes_path, 'r') as classes_file:
            # Set the names of the classes
            classes = [i.split('\n')[0] for i in classes_file.readlines()]

        return classes

    def write_data_yaml(self,
                        anns_type: str) -> None:
        # Read classes file
        classes = DatasetCreator.read_classes_file(
            classes_path=self.classes_path)
        print(f"Available classes is {classes}")

        # Write the data.yaml file
        with open(self.data_yaml_path, 'w', encoding="utf-8") as yaml_file:
            path = os.path.normpath(self.dataset_dir)
            yaml_file.write('path: ' + path + '\n')
            yaml_file.write('train: ' + "train" + '\n')
            yaml_file.write('val: ' + "val" + '\n')
            yaml_file.write('test: ' + "test" + '\n')

            yaml_file.write('names:' + '\n')
            for i, class_name in enumerate(classes):
                yaml_file.write(f"    {i}: {class_name}\n")

            if anns_type == "keypoint":
                yaml_file.write("kpt_shape: [30, 2]")

    @staticmethod
    def copy_files_from_dict(key,
                             value,
                             images_path,
                             labels_path,
                             copy_to) -> None:

        shutil.copyfile(os.path.join(images_path, key),
                        os.path.join(copy_to, key))
        if value is not None:
            shutil.copyfile(os.path.join(labels_path, value),
                            os.path.join(copy_to, value))

    def partitionate_data(self):
        # Dict with images and labels
        data = self.images_labels_dict

        # Create the train, validation, and test datasets
        num_train = int(len(data) * self.train_split)
        num_val = int(len(data) * self.val_split)
        num_test = int(len(data) * self.test_split)

        train_data = {key: data[key] for key in list(data.keys())[:num_train]}
        val_data = {key: data[key] for key in list(
            data.keys())[num_train:num_train+num_val]}
        test_data = {key: data[key] for key in list(
            data.keys())[num_train+num_val:num_train+num_val+num_test]}

        # Copy the images and labels to the train, validation, and test folders
        for key, value in train_data.items():
            DatasetCreator.copy_files_from_dict(key=key,
                                                value=value,
                                                images_path=self.images_path,
                                                labels_path=self.labels_path,
                                                copy_to=self.train_dir)

        for key, value in val_data.items():
            DatasetCreator.copy_files_from_dict(key=key,
                                                value=value,
                                                images_path=self.images_path,
                                                labels_path=self.labels_path,
                                                copy_to=self.val_dir)

        for key, value in test_data.items():
            DatasetCreator.copy_files_from_dict(key=key,
                                                value=value,
                                                images_path=self.images_path,
                                                labels_path=self.labels_path,
                                                copy_to=self.test_dir)

    def create(self, anns_type: str) -> None:
        # Check if annotation type is supported
        assert anns_type in self.anns_types, f"anns_type must be one of {self.anns_types}."

        # Create annotations
        if anns_type == "keypoint":
            self.anns_creator.create_keypoints()
        print("Dataset is creating...")
        self.partitionate_data()
        print("Train, validation, test dataset has created.")
        self.write_data_yaml(anns_type)
        print("data.yaml file has created.")
