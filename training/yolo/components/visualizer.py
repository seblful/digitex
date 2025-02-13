import os
import random

from PIL import Image, ImageDraw

import numpy as np

from tqdm import tqdm

from modules.handlers import LabelHandler

from utils import get_random_image


class Visualizer:
    def __init__(self,
                 dataset_dir: str,
                 check_images_dir: str) -> None:
        # Paths
        self.dataset_dir = dataset_dir
        self.__setup_dataset_dirs()
        self.check_images_dir = check_images_dir

        self.anns_types = ["polygon", "obb"]

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

        return points_dict

    def visualize(self,
                  anns_type: str,
                  num_images: int = 10) -> None:
        assert anns_type in self.label_types, f"label_type must be one of {self.label_types}."

        for set_name, set_dir in self.dataset_dirs.items():
            images_listdir = [image_name for image_name in os.listdir(
                set_dir) if image_name.endswith(".jpg")]

            for _ in tqdm(range(num_images), desc=f"Augmenting {set_name} images"):
                image_name, image = get_random_image(set_dir, images_listdir)
                image_width, image_height = image.size

                points_dict = self.create_polygon(image_name=image_name,
                                                  set_dir=set_dir,
                                                  image_width=image_width,
                                                  image_height=image_height,
                                                  anns_type=anns_type)
