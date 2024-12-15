import os
import shutil
import json
import random

from urllib.parse import unquote
from PIL import Image

import numpy as np
import cv2


class DatasetCreator():
    def __init__(self,
                 raw_dir: str,
                 dataset_dir: str,
                 train_split: float = 0.8) -> None:
        # Paths
        self.raw_dir = raw_dir
        self.dataset_dir = dataset_dir
        self.__setup_dataset_dirs()

        self.__annotation_dir = None

        # Input dirs
        self.json_path = os.path.join(raw_dir, 'data.json')
        self.classes_path = os.path.join(raw_dir, 'classes.txt')

        # Data split
        self.train_split = train_split
        self.test_split = 1 - self.train_split

        self.__id2label = None
        self.__label2id = None

        self.__images_labels_dict = None

        # Annotation creator
        self.annotation_creator = AnnotationCreator(annotation_dir=self.annotation_dir,
                                                    json_path=self.json_path,
                                                    label2id=self.label2id)

    def __setup_dataset_dirs(self) -> None:
        os.mkdir(self.dataset_dir)

        # Create paths
        images_train_dir = os.path.join(
            self.dataset_dir, "ch4_training_images")
        ann_train_dir = os.path.join(
            self.dataset_dir, "ch4_training_localization_transcription_gt")
        images_test_dir = os.path.join(
            self.dataset_dir, "ch4_test_images")
        ann_test_dir = os.path.join(
            self.dataset_dir, "ch4_test_localization_transcription_gt")

        # Create list of dirs
        self.train_dirs = [images_train_dir, ann_train_dir]
        self.test_dirs = [images_test_dir, ann_test_dir]

        # Mkdirs
        for i in range(len(self.train_dirs)):
            os.mkdir(self.train_dirs[i])
            os.mkdir(self.test_dirs[i])

    @property
    def annotation_dir(self) -> str:
        if self.__annotation_dir is None:
            annotation_dir = os.path.join(self.raw_dir, "annotations")
            os.mkdir(annotation_dir)
            self.__annotation_dir = annotation_dir

        return self.__annotation_dir

    @property
    def images_labels_dict(self) -> dict:
        if self.__images_labels_dict is None:
            self.__images_labels_dict = self.__create_images_labels_dict()

        return self.__images_labels_dict

    def __create_images_labels_dict(self, shuffle=True) -> dict:
        # List of all images and labels in directory
        # images = os.listdir(self.images_dir)
        labels = os.listdir(self.annotation_dir)

        # Create a dictionary to store the images and labels names
        images_labels = {}
        for label in labels:
            image = os.path.splitext(label)[0] + '.jpg'

            images_labels[image] = label

        if shuffle:
            # Shuffle the data
            keys = list(images_labels.keys())
            random.shuffle(keys)
            images_labels = {key: images_labels[key] for key in keys}

        return images_labels

    @property
    def id2label(self) -> dict[int, str]:
        if self.__id2label is None:
            self.__id2label = self.__create_id2label()

        return self.__id2label

    @property
    def label2id(self) -> dict[str, int]:
        if self.__label2id is None:
            self.__label2id = {v: k for k, v in self.id2label.items()}

        return self.__label2id

    def __create_id2label(self) -> dict[int, str]:
        with open(self.classes_path, 'r') as file:
            # Set the names of the classes
            classes = [i.split('\n')[0] for i in file.readlines()]
            id2label = {k: v for k, v in enumerate(classes, start=0)}

        return id2label

    def __copy_data(self,
                    data_dict: dict[str, str],
                    dirs_name: str) -> None:

        for image_name, label_name in data_dict.items():
            shutil.copyfile(os.path.join(self.raw_dir, "images", image_name),
                            os.path.join(dirs_name[0], image_name))
            shutil.copyfile(os.path.join(self.annotation_dir, label_name),
                            os.path.join(dirs_name[1], label_name))

    def __partitionate_data(self) -> None:
        # Dict with images and labels
        data = self.images_labels_dict

        # Create the train, validation, and test datasets
        num_train = int(len(data) * self.train_split)
        num_test = int(len(data) * self.test_split)

        # Create dicts with images and labels names
        train_data = {key: data[key] for key in list(data.keys())[:num_train]}
        test_data = {key: data[key] for key in list(
            data.keys())[num_train:num_train+num_test]}

        # Copy the images and labels to the train, validation, and test folders
        for data_dict, dirs_name in zip((train_data, test_data), (self.train_dirs, self.test_dirs)):
            self.__copy_data(data_dict=data_dict,
                             dirs_name=dirs_name)

    def create_dataset(self) -> None:
        # Create annotations
        print("Annotations are creating...")
        # self.annotation_creator.create_annotations()

        # Create dataset
        print("Data is partitioning...")
        self.__partitionate_data()

        print("Train, validation, test sets have created.")
