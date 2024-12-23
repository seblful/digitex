import os
import random

import json
from PIL import Image, ImageDraw


class Visualizer:
    def __init__(self,
                 dataset_dir,
                 check_images_dir) -> None:
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")

        self.check_images_dir = check_images_dir

    @staticmethod
    def read_json(json_path) -> dict:
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_dict = json.load(json_file)

        return json_dict

    @staticmethod
    def _draw_image(image: Image,
                    polygons: list[tuple[int, int]]) -> Image:
        draw = ImageDraw.Draw(image, 'RGBA')
        for polygon in polygons:
            draw.polygon(polygon,
                         fill=((0, 255, 0, 128)),
                         outline="red")

        return image

    @staticmethod
    def _read_ann_file(ann_path: str) -> list[str]:
        with open(ann_path, 'r', encoding="utf-8") as ann_file:
            anns = ann_file.readlines()

        return anns

    @staticmethod
    def __get_polygons(anns_dict: dict,
                       image_name: str) -> list[tuple[int, int]]:
        polygons = anns_dict[image_name]["polygons"]
        polygons = [[tuple(points) for points in polygon]
                    for polygon in polygons]

        return polygons

    def visualize(self, num_images: int = 10) -> None:
        print("Visualizing dataset images...")

        for set_dir in (self.train_dir, self.val_dir):
            images_dir = os.path.join(set_dir, "images")
            images_listdir = os.listdir(images_dir)
            random.shuffle(images_listdir)
            images_listdir = images_listdir[:num_images]

            anns_json_path = os.path.join(set_dir, "labels.json")
            anns_dict = self.read_json(anns_json_path)

            for image_name in images_listdir:
                # Open original image
                image_path = os.path.join(images_dir, image_name)
                image = Image.open(image_path)

                # Get polygons and draw on image
                polygons = self.__get_polygons(anns_dict=anns_dict,
                                               image_name=image_name)
                drawn_image = self._draw_image(image=image,
                                               polygons=polygons)

                self.__save_image(image=drawn_image,
                                  set_dir=set_dir,
                                  image_name=image_name)

    def __save_image(self,
                     image: Image,
                     image_name: str,
                     set_dir: str) -> None:
        image_name = os.path.splitext(image_name)[0]
        set_dir_name = ["train", "val"]["train" != os.path.basename(set_dir)]
        save_image_name = f"{image_name}_{set_dir_name}.jpg"
        save_path = os.path.join(self.check_images_dir, save_image_name)
        image.save(save_path)
