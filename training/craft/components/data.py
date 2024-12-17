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

        self.__images_labels_dict = None

        # Annotation creator
        self.annotation_creator = AnnotationCreator(annotation_dir=self.annotation_dir,
                                                    json_path=self.json_path)

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

    def __copy_data(self,
                    data_dict: dict[str, str],
                    dirs_name: str) -> None:

        counter = 0

        for image_name, label_name in data_dict.items():
            counter += 1
            shutil.copyfile(os.path.join(self.raw_dir, "images", image_name),
                            os.path.join(dirs_name[0], f"img_{str(counter)}.jpg"))
            shutil.copyfile(os.path.join(self.annotation_dir, label_name),
                            os.path.join(dirs_name[1], f"gt_img_{str(counter)}.txt"))

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
        self.annotation_creator.create_annotations()

        # Create dataset
        print("Data is partitioning...")
        self.__partitionate_data()


class AnnotationCreator:
    def __init__(self,
                 annotation_dir: str,
                 json_path: str) -> None:
        # Paths
        self.annotation_dir = annotation_dir
        self.json_path = json_path

    @staticmethod
    def read_json(json_path) -> dict:
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_dict = json.load(json_file)

        return json_dict

    @staticmethod
    def bbox_to_polygon(bbox: list[float],
                        image_width: int,
                        image_height: int) -> list[int]:

        x_rel, y_rel, width_rel, height_rel = bbox

        # Convert relative coordinates to absolute
        x = int((x_rel / 100) * image_width)
        y = int((y_rel / 100) * image_height)
        width = int((width_rel / 100) * image_width)
        height = int((height_rel / 100) * image_height)

        # Define the vertices of the polygon
        polygon = [x, y, x + width, y, x + width, y + height, x, y + height]

        return polygon

    @staticmethod
    def polygon_to_abs(polygon,
                       image_width: int,
                       image_height: int) -> list[int]:

        abs_polygon = []

        for pol in polygon:
            x = int((pol[0] / 100) * image_width)
            y = int((pol[1] / 100) * image_height)

            abs_polygon.append(x)
            abs_polygon.append(y)

        return abs_polygon

    def __get_polygons(self, task) -> tuple[list, list]:
        # Create empty arrays to store data
        texts = []
        polygons = []

        # Retrieve result
        result = task['annotations'][0]['result']

        for entry in result:
            if entry['type'] == 'textarea':
                # Extract text details
                text = entry['value']['text'][0]
                texts.append(text)

                # Extract bbox and convert it to polygon with absolute coordinates
                if 'x' in entry['value'].keys():
                    bbox = [entry['value']['x'],
                            entry['value']['y'],
                            entry['value']['width'],
                            entry['value']['height']]

                    polygon = self.bbox_to_polygon(bbox=bbox,
                                                   image_width=entry['original_width'],
                                                   image_height=entry['original_height'])
                    polygons.append(polygon)

                # Extract polygon and convert it to polygon with absolute coordinates
                elif 'points' in entry['value'].keys():
                    polygon = entry['value']['points']
                    polygon = self.polygon_to_abs(polygon=polygon,
                                                  image_width=entry['original_width'],
                                                  image_height=entry['original_height'])

                    polygons.append(polygon)

        return polygons, texts

    def create_annotations(self) -> None:
        # Read json_polygon_path
        json_dict = self.read_json(self.json_path)

        # Iterating through tasks
        for task in json_dict:

            # Get polygons and labels
            polygons, texts = self.__get_polygons(task)

            # Save annotation as image
            image_name = unquote(os.path.basename(task["data"]["image"]))
            self.__save_annotation(polygons=polygons,
                                   texts=texts,
                                   image_name=image_name)

    def __save_annotation(self,
                          polygons: list[int],
                          texts: list[str],
                          image_name: str) -> None:
        # Create annotation path
        annotation_name = os.path.splitext(image_name)[0] + ".txt"
        annotation_path = os.path.join(self.annotation_dir, annotation_name)

        with open(annotation_path, 'a', encoding="utf-8") as file:
            for i in range(len(polygons)):
                str_to_write = list(map(str, polygons[i])) + [texts[i]]
                file.write(",".join(str_to_write))
                file.write("\n")

        return None
