from typing import LiteralString, List, Dict

import os
import shutil
import random


class DatasetCreator:
    def __init__(self,
                 raw_dir,
                 dataset_dir,
                 train_split=0.8) -> None:

        self.raw_dir = raw_dir
        self.dataset_dir = dataset_dir

        self.__images_path = None  # Raw images
        self.__labels_path = None  # Raw labels

        self.__train_folder = None
        self.__val_folder = None
        self.__test_folder = None

        self.__images_labels_dict = None
        self.__classes_path = None
        self.__data_yaml_path = None

        self.train_split = train_split
        self.val_split = 0.6 * (1 - self.train_split)
        self.test_split = 1 - self.train_split - self.val_split

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
    def train_folder(self) -> LiteralString | str:
        if self.__train_folder is None:
            train_folder = os.path.join(self.dataset_dir, 'train')
            # Creating folder for train set
            os.makedirs(train_folder, exist_ok=True)
            self.__train_folder = train_folder

        return self.__train_folder

    @property
    def val_folder(self) -> LiteralString | str:
        if self.__val_folder is None:
            val_folder = os.path.join(self.dataset_dir, 'val')
            # Creating folder for val set
            os.makedirs(val_folder, exist_ok=True)
            self.__val_folder = val_folder

        return self.__val_folder

    @property
    def test_folder(self) -> LiteralString | str:
        if self.__test_folder is None:
            test_folder = os.path.join(self.dataset_dir, 'test')
            # Creating folder for test set
            os.makedirs(test_folder, exist_ok=True)
            self.__test_folder = test_folder

        return self.__test_folder

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

    def write_data_yaml(self):
        # Read classes file
        classes = DatasetCreator.read_classes_file(
            classes_path=self.classes_path)
        print(f"Available classes is {classes}")

        # Write the data.yaml file
        with open(self.data_yaml_path, 'w', encoding="utf-8") as yaml_file:
            yaml_file.write('names:' + '\n')
            for class_name in classes:
                yaml_file.write(f"- {class_name}\n")

            yaml_file.write('nc: ' + str(len(classes)) + '\n')

            yaml_file.write('train: ' + self.train_folder + '\n')
            yaml_file.write('val: ' + self.val_folder + '\n')
            yaml_file.write('test: ' + self.test_folder + '\n')

    @staticmethod
    def copy_files_from_dict(key,
                             value,
                             images_path,
                             labels_path,
                             copy_to):

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
                                                copy_to=self.train_folder)

        for key, value in val_data.items():
            DatasetCreator.copy_files_from_dict(key=key,
                                                value=value,
                                                images_path=self.images_path,
                                                labels_path=self.labels_path,
                                                copy_to=self.val_folder)

        for key, value in test_data.items():
            DatasetCreator.copy_files_from_dict(key=key,
                                                value=value,
                                                images_path=self.images_path,
                                                labels_path=self.labels_path,
                                                copy_to=self.test_folder)

    def process(self):
        # Create train, valid, test datasets
        print("Dataset is creating...")
        self.partitionate_data()
        print("Train, validation, test dataset has created.")
        self.write_data_yaml()
        print("data.yaml file has created.")
