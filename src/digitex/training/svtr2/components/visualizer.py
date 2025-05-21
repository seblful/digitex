import os
import random
from typing import Optional
from abc import ABC, abstractmethod

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from digitex.core.processors.file import FileProcessor

import lmdb
import io


class BaseVisualizer(ABC):
    def __init__(
        self, dataset_dir: str, check_images_dir: str, font_path: Optional[str] = None
    ) -> None:
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.dataset_dirs = [self.train_dir, self.val_dir]
        self.check_images_dir = check_images_dir

        self.font_path = font_path
        self.font_size = 30
        self._font: Optional[ImageFont.ImageFont] = None

    @property
    def font(self) -> ImageFont.ImageFont:
        if self._font is None:
            try:
                self._font = ImageFont.truetype(
                    self.font_path, self.font_size, encoding="unic"
                )
            except Exception:
                self._font = ImageFont.load_default()
        return self._font

    def _draw_image(
        self, image: Image.Image, label: str, padding: int = 15
    ) -> Image.Image:
        text_bbox = self.font.getbbox(label, anchor="mt")
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        new_width = max(image.width, text_width + 2 * padding)
        new_height = image.height + text_height + 2 * padding

        drawn_image = Image.new("RGB", (new_width, new_height), "white")
        drawn_image.paste(image, ((new_width - image.width) // 2, 0))

        draw = ImageDraw.Draw(drawn_image)

        text_x = (new_width - text_width) // 2
        text_y = image.height + padding // 2
        draw.text((text_x, text_y), label, font=self.font, fill="red")
        return drawn_image

    @staticmethod
    def _find_dir_name(set_dir: str) -> str:
        basename = os.path.basename(set_dir)
        if basename in ["train", "real"]:
            return "train"
        elif basename == "val":
            return "val"
        return "test"

    def _save_image(self, set_dir: str, image: Image.Image, idx: int) -> None:
        set_dir_name = self._find_dir_name(set_dir)
        image_path = f"{idx}_{set_dir_name}.jpg"
        save_path = os.path.join(self.check_images_dir, image_path)
        image.save(save_path)

    @abstractmethod
    def visualize(self, num_images: int = 10) -> None:
        pass


class SimpleVisualizer(BaseVisualizer):
    def _get_data(self, set_dir: str) -> tuple[list[str], list[str]]:
        image_paths, texts = [], []
        labels_txt_path = os.path.join(set_dir, "labels.txt")
        lines = FileProcessor.read_txt(labels_txt_path, strip=True)
        for line in lines:
            dst_image_path, text = line.split("\t", maxsplit=1)
            image_path = os.path.join(set_dir, dst_image_path)
            image_paths.append(image_path)
            texts.append(text)
        return image_paths, texts

    def visualize(self, num_images: int = 10) -> None:
        for set_dir in self.dataset_dirs:
            dir_name = os.path.basename(set_dir)
            image_paths, texts = self._get_data(set_dir)
            data_len = len(image_paths)
            for counter in tqdm(
                range(num_images), desc=f"Visualizing {dir_name} images"
            ):
                rnd_idx = random.randint(0, data_len - 1)
                image_path, text = image_paths[rnd_idx], texts[rnd_idx]
                with Image.open(image_path) as image:
                    drawn_image = self._draw_image(image=image, label=text)
                    self._save_image(set_dir=set_dir, image=drawn_image, idx=counter)
