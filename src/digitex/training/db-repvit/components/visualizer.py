import os
import json
import random

from PIL import Image, ImageDraw

from digitex.core.processors.file import FileProcessor


class Visualizer:
    def __init__(self, dataset_dir, check_images_dir) -> None:
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")

        self.check_images_dir = check_images_dir

    @staticmethod
    def _draw_image(image: Image, polygons: list[tuple[int, int]]) -> Image:
        draw = ImageDraw.Draw(image, "RGBA")
        for polygon in polygons:
            draw.polygon(polygon, fill=((0, 255, 0, 128)), outline="red")

        return image

    @staticmethod
    def _get_polygons(anns_str: str) -> list[tuple[int, int]]:
        anns_str = anns_str.replace("'", '"')
        anns = json.loads(anns_str)
        polygons = []

        for ann in anns:
            points = ann["points"]
            polygons.append(tuple(points))

        return polygons

    def visualize(self, num_images: int = 10) -> None:
        print("Visualizing dataset images...")

        for set_dir in (self.train_dir, self.val_dir):
            images_dir = os.path.join(set_dir, "images")

            anns_txt_path = os.path.join(set_dir, "labels.txt")
            anns_lines = FileProcessor.read_txt(anns_txt_path)
            random.shuffle(anns_lines)
            anns_lines = anns_lines[:num_images]

            for anns_line in anns_lines:
                image_name, anns_str = anns_line.split("\t", 1)
                image_path = os.path.join(images_dir, image_name)
                image = Image.open(image_path)
                polygons = self._get_polygons(anns_str)

                # Draw and save image
                drawn_image = self._draw_image(image=image, polygons=polygons)
                self._save_image(
                    image=drawn_image, set_dir=set_dir, image_name=image_name
                )

    def _save_image(self, image: Image, image_name: str, set_dir: str) -> None:
        image_name = os.path.splitext(image_name)[0]
        set_dir_name = ["train", "val"]["train" != os.path.basename(set_dir)]
        save_image_name = f"{image_name}_{set_dir_name}.jpg"
        save_path = os.path.join(self.check_images_dir, save_image_name)
        image.save(save_path)
