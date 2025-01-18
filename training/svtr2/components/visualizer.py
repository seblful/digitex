import os
import random

from PIL import Image, ImageDraw, ImageFont

from tqdm import tqdm


class Visualizer:
    def __init__(self,
                 dataset_dir,
                 check_images_dir) -> None:
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.dataset_dirs = [self.train_dir, self.val_dir]

        self.check_images_dir = check_images_dir

    @staticmethod
    def read_txt(txt_path: str) -> list[str]:
        with open(txt_path, 'r', encoding="utf-8") as ann_file:
            lines = ann_file.readlines()

        return lines

    def _get_data(self, set_dir) -> tuple[list[str], list[Image.Image]]:
        # Create empty lists to store image paths and texts
        image_paths = []
        texts = []

        # Get gt_lines
        gt_txt_path = os.path.join(set_dir, "gt.txt")
        gt_lines = self.read_txt(gt_txt_path)

        # Iterate through each line
        for gt_line in gt_lines:
            image_name, text = gt_line.split(maxsplit=1)

            # Preprocess image and text
            image_path = os.path.join(set_dir, "images", image_name)

            text = text.strip()

            image_paths.append(image_path)
            texts.append(text)

        return image_paths, texts

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

    def visualize(self, num_images: int = 10) -> None:
        for set_dir in self.dataset_dirs:
            dir_name = os.path.basename(set_dir)
            # Get data from gt
            image_paths, texts = self._get_data(set_dir)
            data_len = len(image_paths)
            
            for counter in tqdm(range(num_images), desc=f"Visualizing {dir_name} images"):
                # Get random image, draw text on it and save
                rnd_idx = random.randint(0, data_len-1)
                image_path, text = image_paths[rnd_idx], texts[rnd_idx]
                image = Image.open(image_path)
                drawn_image = self._draw_image(image=image, label=text)

                self._save_image(set_dir=set_dir,
                            image=drawn_image,
                            idx=counter)

                image.close()

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
