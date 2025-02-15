import os
import random

from PIL import Image, ImageDraw

import numpy as np

from tqdm import tqdm

from modules.handlers import LabelHandler

from .converter import Converter
from .utils import get_random_image


class Visualizer:
    def __init__(self,
                 dataset_dir: str,
                 check_images_dir: str) -> None:
        # Paths
        self.dataset_dir = dataset_dir
        self.__setup_dataset_dirs()
        self.check_images_dir = check_images_dir

        self.anns_types = ["polygon", "obb"]
        self.preprocess_funcs = {"polygon": Converter.point_to_polygon,
                                 "obb": Converter.xyxyxyxy_to_polygon}

        self.label_handler = LabelHandler()

        self.colors = {0: (255, 0, 0, 128),
                       1: (0, 255, 0, 128),
                       2: (0, 0, 255, 128),
                       3: (255, 255, 0, 128),
                       4: (255, 0, 255, 128),
                       5: (0, 255, 255, 128),
                       6: (128, 0, 128, 128),
                       7: (255, 165, 0, 128)}

    def __setup_dataset_dirs(self) -> None:
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.val_dir = os.path.join(self.dataset_dir, 'val')
        self.test_dir = os.path.join(self.dataset_dir, 'test')

        self.dataset_dirs = {"train": self.train_dir,
                             "val": self.val_dir,
                             "test": self.test_dir}

    def draw_image(self,
                   image: Image.Image,
                   polygons_dict: dict[int, list]) -> Image:
        if polygons_dict is None:
            return image

        draw = ImageDraw.Draw(image, 'RGBA')
        for class_idx, polygons in polygons_dict.items():
            color = self.colors[class_idx]
            for polygon in polygons:
                draw.polygon(polygon,
                             fill=color,
                             outline="black")

        return image

    def save_image(self,
                   image: Image,
                   image_name: str,
                   set_name: str) -> None:
        name = os.path.splitext(image_name)[0]
        filename = f"{name}_{set_name}.jpg"
        filepath = os.path.join(self.check_images_dir, filename)
        image.save(filepath)

    def create_polygon(self,
                       image_name: str,
                       set_dir: str,
                       image_width: str,
                       image_height: str,
                       anns_type: str) -> None | dict[int, list]:
        anns_name = os.path.splitext(image_name)[0] + '.txt'
        anns_path = os.path.join(set_dir, anns_name)
        points_dict = self.label_handler._read_points(anns_path)

        if not points_dict:
            return None

        polygons_dict = {key: [] for key in points_dict.keys()}
        preprocess_func = self.preprocess_funcs[anns_type]

        for class_idx, points in points_dict.items():
            for point in points:
                polygon = preprocess_func(
                    point, image_width, image_height).tolist()
                polygon = [tuple(row) for row in polygon]
                polygons_dict[class_idx].append(polygon)

        return polygons_dict

    def visualize(self,
                  anns_type: str,
                  num_images: int = 10) -> None:
        assert anns_type in self.anns_types, f"label_type must be one of {self.anns_types}."

        for set_name, set_dir in self.dataset_dirs.items():
            images_listdir = [image_name for image_name in os.listdir(
                set_dir) if image_name.endswith(".jpg")]
            random.shuffle(images_listdir)
            images_listdir = images_listdir[:num_images]

            for i in tqdm(range(len(images_listdir)), desc=f"Visualizing {set_name} images"):
                image_name = images_listdir[i]
                image_path = os.path.join(set_dir, image_name)
                image = Image.open(image_path)
                image_width, image_height = image.size

                polygons_dict = self.create_polygon(image_name=image_name,
                                                    set_dir=set_dir,
                                                    image_width=image_width,
                                                    image_height=image_height,
                                                    anns_type=anns_type)

                drawn_image = self.draw_image(image=image,
                                              polygons_dict=polygons_dict)
                self.save_image(image=drawn_image,
                                image_name=image_name,
                                set_name=set_name)
