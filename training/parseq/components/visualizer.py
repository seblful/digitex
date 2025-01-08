import os
import random
import io
from PIL import Image, ImageDraw, ImageFont

import lmdb
import numpy as np


class Visualizer:
    def __init__(self,
                 dataset_dir,
                 check_images_dir,
                 source: str) -> None:
        self.dataset_dir = dataset_dir
        train_subdir = "real" if source == "ls" else "synth"
        self.train_dir = os.path.join(self.dataset_dir, "train", train_subdir)
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        self.dataset_dirs = [self.train_dir, self.val_dir, self.test_dir]

        self.check_images_dir = check_images_dir

    @staticmethod
    def _read_lmdb(set_dir) -> lmdb.Environment:
        env = lmdb.open(set_dir, readonly=True)
        return env

    @staticmethod
    def _draw_image(image: Image.Image,
                    label: str,
                    font_size: int = 30,
                    padding=15) -> Image.Image:
        font = ImageFont.truetype("arial.ttf", font_size)

        # Get size of text
        dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        text_bbox = dummy_draw.textbbox((0, 0), label, font=font, anchor="mt")
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Create new image with space for text
        new_width = max(image.width, text_width + 2 * padding)
        new_height = image.height + text_height + 2 * padding
        drawn_image = Image.new('RGB', (new_width, new_height), 'white')

        # Paste original image
        drawn_image.paste(image, ((new_width - image.width) // 2, 0))

        # Draw text
        draw = ImageDraw.Draw(drawn_image)
        text_x = (new_width - text_width) // 2
        text_y = image.height + padding // 2
        draw.text((text_x, text_y), label, font=font, fill='red')

        return drawn_image

    def _get_data(self, set_dir) -> tuple[list[str], list[Image.Image]]:
        env = self._read_lmdb(set_dir)

        with env.begin() as txn:
            # Get number of samples
            n_samples = int(txn.get('num-samples'.encode()).decode())

            # Create lists to store data
            images = []
            labels = []

            # Read all samples
            for idx in range(1, n_samples + 1):
                # Construct keys
                image_key = f'image-{idx:09d}'.encode()
                label_key = f'label-{idx:09d}'.encode()

                # Get data
                image_bytes = txn.get(image_key)
                image_buffer = io.BytesIO(image_bytes)
                image = Image.open(image_buffer)
                label = txn.get(label_key).decode()

                images.append(image)
                labels.append(label)

        return images, labels

    def visualize(self, num_images: int = 10) -> None:
        print("Visualizing dataset images...")

        for set_dir in (self.dataset_dirs):
            counter = 0

            # Get data from lmdb database
            images, labels = self._get_data(set_dir)
            data_len = len(images)

            while counter < num_images:
                # Get random image, draw text on it and save
                rnd_idx = random.randint(0, data_len-1)
                image, label = images[rnd_idx], labels[rnd_idx]
                drawn_image = self._draw_image(image=image, label=label)

                self._save_image(set_dir=set_dir,
                                 image=drawn_image,
                                 idx=counter)

                counter += 1

    @staticmethod
    def _find_dir_name(set_dir: str):
        basename = os.path.basename(set_dir)

        if basename in ["train", "real"]:
            return "train"
        elif basename in ["val"]:
            return "val"
        else:
            return "test"

    def _save_image(self,
                    set_dir: str,
                    image: Image,
                    idx: int) -> None:
        set_dir_name = self._find_dir_name(set_dir)
        image_path = f"{str(idx)}_{set_dir_name}.jpg"
        save_path = os.path.join(self.check_images_dir, image_path)
        image.save(save_path)
