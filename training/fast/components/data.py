import os
import shutil
import json

import math
import random

import hashlib
from urllib.parse import unquote


class DatasetCreator():
    def __init__(self,
                 raw_dir: str,
                 dataset_dir: str,
                 train_split: float = 0.8) -> None:
        # Paths
        self.raw_dir = raw_dir
        self.raw_images_dir = os.path.join(raw_dir, "images")
        self.anns_json_path = os.path.join(raw_dir, "anns.json")

        self.dataset_dir = dataset_dir
        self.__setup_dataset_dirs()

        # Input dirs
        self.data_json_path = os.path.join(raw_dir, 'data.json')
        self.classes_path = os.path.join(raw_dir, 'classes.txt')

        # Data split
        self.train_split = train_split
        self.val_split = 1 - self.train_split

        # Annotation creator
        self.annotation_creator = AnnotationCreator(raw_images_dir=self.raw_images_dir,
                                                    data_json_path=self.data_json_path,
                                                    anns_json_path=self.anns_json_path)

    def __setup_dataset_dirs(self) -> None:
        os.mkdir(self.dataset_dir)

        # Train dirs
        self.train_dir = os.path.join(self.dataset_dir, "train")
        train_images_dir = os.path.join(self.train_dir, "images")
        os.makedirs(train_images_dir)

        # Val dirs
        self.val_dir = os.path.join(self.dataset_dir, "val")
        val_images_dir = os.path.join(self.val_dir, "images")
        os.makedirs(val_images_dir)

    def __copy_data(self,
                    listdir: list[str],
                    set_dir: str,
                    anns_dict: dict) -> None:
        # Create empty dict to store formatted annotations
        set_anns_dict = {}

        # Copy image
        for image_name in listdir:
            shutil.copyfile(os.path.join(self.raw_images_dir, image_name),
                            os.path.join(set_dir, "images", image_name))
            set_anns_dict[image_name] = anns_dict[image_name]

        # Save annotation dict
        json_path = os.path.join(set_dir, "labels.json")
        self.annotation_creator.write_json(json_dict=set_anns_dict,
                                           json_path=json_path)

    def __partitionate_data(self) -> None:
        # Images listdir and shuffle
        images_listdir = os.listdir(self.raw_images_dir)
        random.shuffle(images_listdir)

        # Create train and validation listdirs
        num_train = int(len(images_listdir) * self.train_split)
        num_val = int(len(images_listdir) * self.val_split)
        train_listdir = images_listdir[:num_train]
        val_listdir = images_listdir[num_train:num_train+num_val]

        # Load anns dict
        anns_dict = self.annotation_creator.read_json(
            json_path=self.anns_json_path)

        # Copy the images to folders and create annotation file
        for listdir, set_dir in zip((train_listdir, val_listdir), (self.train_dir, self.val_dir)):
            self.__copy_data(listdir=listdir,
                             set_dir=set_dir,
                             anns_dict=anns_dict)

    def create_dataset(self) -> None:
        # Create annotations
        print("Annotations are creating...")
        self.annotation_creator.create_annotations()

        # Create dataset
        print("Data is partitioning...")
        self.__partitionate_data()


class AnnotationCreator:
    def __init__(self,
                 raw_images_dir: str,
                 data_json_path: str,
                 anns_json_path: str) -> None:
        # Paths
        self.raw_images_dir = raw_images_dir
        self.data_json_path = data_json_path
        self.anns_json_path = anns_json_path

    @staticmethod
    def read_json(json_path) -> dict:
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_dict = json.load(json_file)

        return json_dict

    @staticmethod
    def write_json(json_dict: dict,
                   json_path: str,
                   indent: int = 4) -> None:
        with open(json_path, 'w', encoding="utf-8") as json_file:
            json.dump(json_dict, json_file,
                      indent=indent,
                      ensure_ascii=False)

        return None

    def image_hash(self,
                   image_name: str) -> str:
        image_path = os.path.join(self.raw_images_dir, image_name)
        with open(image_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()

        return hash

    @staticmethod
    def bbox_to_polygon(bbox: dict,
                        image_width: int,
                        image_height: int) -> list[tuple[int, int]]:

        x_rel, y_rel, width_rel, height_rel, rotation = bbox["x"], bbox[
            "y"], bbox["width"], bbox["height"], bbox["rotation"]

        # Convert relative coordinates to absolute
        x = (x_rel / 100) * image_width
        y = (y_rel / 100) * image_height
        width = (width_rel / 100) * image_width
        height = (height_rel / 100) * image_height

        # Define the vertices of the rectangle (unrotated)
        vertices = [(x, y), (x + width, y),
                    (x + width, y + height), (x, y + height)]

        # Convert rotation to radians
        angle_rad = math.radians(rotation)

        # Define the rotation matrix
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        # Rotate each vertex around the top-left corner of the bounding box
        rotated_polygon = []
        for vx, vy in vertices:
            rx = x + (vx - x) * cos_theta - (vy - y) * sin_theta
            ry = y + (vx - x) * sin_theta + (vy - y) * cos_theta
            rotated_polygon.append((int(rx), int(ry)))

        return rotated_polygon

    @staticmethod
    def polygon_to_abs(polygon,
                       image_width: int,
                       image_height: int) -> None:

        for i in range(len(polygon)):
            polygon[i][0] = int((polygon[i][0] / 100) * image_width)
            polygon[i][1] = int((polygon[i][1] / 100) * image_height)

        return None

    def __get_polygons(self, task) -> tuple[list, int, int]:
        # Create empty array to store data
        polygons = []

        # Retrieve result
        result = task['annotations'][0]['result']

        for entry in result:
            if entry['type'] == 'textarea':
                # Width and height
                image_width = entry["original_width"]
                image_height = entry["original_height"]

                # Extract bbox and convert it to polygon with absolute coordinates
                if 'x' in entry['value'].keys():
                    bbox = {k: v for k,
                            v in entry['value'].items() if k != "text"}

                    polygon = self.bbox_to_polygon(bbox=bbox,
                                                   image_width=image_width,
                                                   image_height=image_height)
                    polygons.append(polygon)

                # Extract polygon and convert it to polygon with absolute coordinates
                elif 'points' in entry['value'].keys():
                    polygon = entry['value']['points']
                    self.polygon_to_abs(polygon=polygon,
                                        image_width=image_width,
                                        image_height=image_height)
                    polygon = [tuple(points) for points in polygon]

                    polygons.append(polygon)

        return polygons, image_width, image_height

    def create_annotations(self) -> None:
        # Read json_polygon_path
        json_dict = self.read_json(self.data_json_path)

        # Create empty dict to store formatted annotations
        anns_dict = {}

        # Iterating through tasks
        for task in json_dict:

            # Get polygons and labels
            polygons, image_width, image_height = self.__get_polygons(
                task)

            # Fill ann dict
            image_name = unquote(os.path.basename(task["data"]["image"]))
            anns_dict[image_name] = {"img_dimensions": (image_width, image_height),
                                     "img_hash": self.image_hash(image_name),
                                     "polygons": polygons}

        # Save annotation
        self.write_json(json_dict=anns_dict,
                        json_path=self.anns_json_path)
