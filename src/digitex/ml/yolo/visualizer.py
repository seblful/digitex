import logging
import random
from pathlib import Path

from PIL import Image, ImageDraw

from tqdm import tqdm

from digitex.core.handlers import LabelHandler

from .converter import Converter

logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self,
                 dataset_dir: str | Path,
                 check_images_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.check_images_dir = Path(check_images_dir)

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
        self.train_dir = self.dataset_dir / "train"
        self.val_dir = self.dataset_dir / "val"
        self.test_dir = self.dataset_dir / "test"

        self.dataset_dirs = {
            "train": self.train_dir,
            "val": self.val_dir,
            "test": self.test_dir,
        }

    def save_image(
        self, image: Image.Image, image_name: str, set_name: str
    ) -> None:
        name = Path(image_name).stem
        filename = f"{name}_{set_name}.jpg"
        filepath = self.check_images_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(filepath))

    def visualize(self, num_images: int = 10) -> None:
        for set_name, set_dir in self.dataset_dirs.items():
            images = [
                img.name
                for img in set_dir.iterdir()
                if img.suffix == ".jpg"
            ]
            random.shuffle(images)
            selected_images = images[:num_images]

            for image_name in tqdm(selected_images, desc=f"Visualizing {set_name}"):
                image_path = set_dir / image_name
                image = Image.open(str(image_path))
                img_width, img_height = image.size

                annotations = self.create_annotations(
                    image_name, set_dir, img_width, img_height
                )
                drawn_image = self.draw_annotations(image, annotations)
                self.save_image(drawn_image, image_name, set_name)

    def create_annotations(self,
                           image_name: str,
                           set_dir: Path,
                           img_width: int,
                           img_height: int):
        raise NotImplementedError(
            "Subclasses must implement create_annotations")

    def draw_annotations(self,
                         image: Image.Image,
                         annotations) -> Image.Image:
        raise NotImplementedError("Subclasses must implement draw_annotations")


class PolygonVisualizer(Visualizer):
    def __init__(self,
                 dataset_dir: str | Path,
                 check_images_dir: str | Path) -> None:
        super().__init__(dataset_dir, check_images_dir)

        self.preprocess_func = Converter.point_to_polygon

    def create_annotations(
        self, image_name: str, set_dir: Path, img_width: int, img_height: int
    ) -> dict | None:
        anns_path = set_dir / f"{Path(image_name).stem}.txt"
        points_dict = LabelHandler._read_points(str(anns_path))

        if not points_dict:
            return None

        polygons = {cls: [] for cls in points_dict}
        for cls, points in points_dict.items():
            for pt in points:
                polygon = self.preprocess_func(
                    pt, img_width, img_height
                ).tolist()
                polygons[cls].append([tuple(p) for p in polygon])

        return polygons

    def draw_annotations(self,
                         image: Image.Image,
                         annotations: dict) -> Image.Image:
        if not annotations:
            return image

        draw = ImageDraw.Draw(image, 'RGBA')
        for cls, polygons in annotations.items():
            for polygon in polygons:
                draw.polygon(polygon, fill=self.colors[cls], outline="black")
        return image
