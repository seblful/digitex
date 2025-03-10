import os
import random
from PIL import Image, ImageDraw

from tqdm import tqdm

from .converter import Converter
from modules.handlers import LabelHandler


class Visualizer:
    def __init__(self, dataset_dir: str, check_images_dir: str) -> None:
        self.dataset_dir = dataset_dir
        self.check_images_dir = check_images_dir
        self.label_handler = LabelHandler()
        self.colors = {
            0: (255, 0, 0, 128),
            1: (0, 255, 0, 128),
            2: (0, 0, 255, 128),
            3: (255, 255, 0, 128),
            4: (255, 0, 255, 128),
            5: (0, 255, 255, 128),
            6: (128, 0, 128, 128),
            7: (255, 165, 0, 128)
        }
        self.__setup_dataset_dirs()

    def __setup_dataset_dirs(self) -> None:
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.val_dir = os.path.join(self.dataset_dir, 'val')
        self.test_dir = os.path.join(self.dataset_dir, 'test')
        self.dataset_dirs = {
            "train": self.train_dir,
            "val": self.val_dir,
            "test": self.test_dir
        }

    def save_image(self, image: Image.Image, image_name: str, set_name: str) -> None:
        name = os.path.splitext(image_name)[0]
        filename = f"{name}_{set_name}.jpg"
        filepath = os.path.join(self.check_images_dir, filename)
        image.save(filepath)

    def visualize(self, num_images: int = 10) -> None:
        for set_name, set_dir in self.dataset_dirs.items():
            images = [img for img in os.listdir(
                set_dir) if img.endswith(".jpg")]
            random.shuffle(images)
            selected_images = images[:num_images]

            for image_name in tqdm(selected_images, desc=f"Visualizing {set_name}"):
                image_path = os.path.join(set_dir, image_name)
                image = Image.open(image_path)
                width, height = image.size

                annotations = self.create_annotations(
                    image_name, set_dir, width, height)
                drawn_image = self.draw_annotations(image, annotations)
                self.save_image(drawn_image, image_name, set_name)

    def create_annotations(self,
                           image_name: str,
                           set_dir: str,
                           width: int,
                           height: int):
        raise NotImplementedError(
            "Subclasses must implement create_annotations")

    def draw_annotations(self,
                         image: Image.Image,
                         annotations) -> Image.Image:
        raise NotImplementedError("Subclasses must implement draw_annotations")


class OBB_PolygonVisualizer(Visualizer):
    def __init__(self,
                 dataset_dir: str,
                 check_images_dir: str,
                 anns_type: str) -> None:
        super().__init__(dataset_dir, check_images_dir)

        self.anns_type = anns_type
        self.preprocess_funcs = {
            "polygon": Converter.point_to_polygon,
            "obb": Converter.xyxyxyxy_to_polygon
        }

        if anns_type not in self.preprocess_funcs:
            raise ValueError(
                f"anns_type must be one of {list(self.preprocess_funcs.keys())}.")

        self.preprocess_func = self.preprocess_funcs[anns_type]

    def create_annotations(self, image_name: str, set_dir: str, width: int, height: int) -> dict:
        anns_path = os.path.join(
            set_dir, os.path.splitext(image_name)[0] + '.txt')
        points_dict = self.label_handler._read_points(anns_path)
        if not points_dict:
            return None

        polygons = {cls: [] for cls in points_dict}
        for cls, points in points_dict.items():
            for pt in points:
                polygon = self.preprocess_func(pt, width, height).tolist()
                polygons[cls].append([tuple(p) for p in polygon])
        return polygons

    def draw_annotations(self, image: Image.Image, annotations: dict) -> Image.Image:
        if not annotations:
            return image

        draw = ImageDraw.Draw(image, 'RGBA')
        for cls, polygons in annotations.items():
            for polygon in polygons:
                draw.polygon(polygon, fill=self.colors[cls], outline="black")
        return image
