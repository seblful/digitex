import os
import json
import random

from PIL import Image, ImageDraw, ImageFont

from digitex.core.processors.file import FileProcessor


class Visualizer:
    def __init__(self, dataset_dir, check_images_dir) -> None:
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.check_images_dir = check_images_dir

    @staticmethod
    def _draw_polygons(draw: ImageDraw.Draw, polygons: list[tuple[int, int]]) -> None:
        for polygon in polygons:
            draw.polygon(polygon, fill=((0, 255, 0, 128)), outline="red")

    @staticmethod
    def _add_transcription_text(
        image: Image, transcriptions: list[str], font_size: int = 30, padding: int = 10
    ) -> Image:
        font = ImageFont.truetype("arial.ttf", size=font_size)
        text = " | ".join(transcriptions)

        # Calculate text wrapping based on image width
        max_text_width = image.width - 2 * padding
        lines = []
        current_line = ""
        for word in text.split():
            test_line = f"{current_line} {word}".strip()
            test_line_width = font.getbbox(test_line)[2]  # Get text width
            if test_line_width <= max_text_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        # Calculate new image height to fit text
        text_height = len(lines) * (font_size + padding)
        new_image = Image.new(
            "RGBA", (image.width, image.height + text_height + padding), "white"
        )
        new_image.paste(image, (0, 0))

        # Draw text on the new image
        draw = ImageDraw.Draw(new_image)
        y_text = image.height + padding
        for line in lines:
            text_width = font.getbbox(line)[2]  # Get text width
            x_text = (image.width - text_width) // 2  # Center align text
            draw.text((x_text, y_text), line, fill="black", font=font)
            y_text += font_size + padding

        return new_image

    def _draw_image(
        self, image: Image, polygons: list[tuple[int, int]], transcriptions: list[str]
    ) -> Image:
        draw = ImageDraw.Draw(image, "RGBA")
        self._draw_polygons(draw, polygons)
        return self._add_transcription_text(image, transcriptions)

    def _load_annotations(self, anns_txt_path: str, num_images: int) -> dict:
        anns_lines = FileProcessor.read_txt(anns_txt_path)
        random.shuffle(anns_lines)
        anns_lines = anns_lines[:num_images]

        anns_dict = {}
        for anns_line in anns_lines:
            image_name, anns_str = anns_line.split("\t", 1)
            anns_dict[image_name] = json.loads(anns_str.replace("'", '"'))
        return anns_dict

    def _process_image(
        self, image_name: str, annotations: list[dict], images_dir: str, set_dir: str
    ) -> None:
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path)

        polygons = [tuple(ann["points"]) for ann in annotations]
        transcriptions = [ann["transcription"] for ann in annotations]

        drawn_image = self._draw_image(
            image=image, polygons=polygons, transcriptions=transcriptions
        )
        self._save_image(image=drawn_image, image_name=image_name, set_dir=set_dir)

    def visualize(self, num_images: int = 10) -> None:
        print("Visualizing dataset images...")

        for set_dir in (self.train_dir, self.val_dir):
            images_dir = os.path.join(set_dir, "images")
            anns_txt_path = os.path.join(set_dir, "labels.txt")
            anns_dict = self._load_annotations(anns_txt_path, num_images)

            for image_path, annotations in anns_dict.items():
                image_name = os.path.basename(image_path)
                self._process_image(image_name, annotations, images_dir, set_dir)

    def _save_image(self, image: Image, image_name: str, set_dir: str) -> None:
        image_name = os.path.splitext(image_name)[0]
        set_dir_name = ["train", "val"]["train" != os.path.basename(set_dir)]
        save_image_name = f"{image_name}_{set_dir_name}.png"
        save_path = os.path.join(self.check_images_dir, save_image_name)
        image.save(save_path)
