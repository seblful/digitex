import os
import random
from PIL import Image, ImageDraw

from tqdm import tqdm

from digitex.core.processors.file import FileProcessor
from digitex.training.superpoint.components.annotation import AbsoluteKeypointsObject
from digitex.training.superpoint.components.augmenter import KeypointAugmenter


class KeypointVisualizer:
    def __init__(self, dataset_dir: str, check_images_dir: str) -> None:
        self.dataset_dir = dataset_dir
        self.check_images_dir = check_images_dir

        self.colors = {
            0: (255, 0, 0, 128),
            1: (0, 255, 0, 128),
            2: (0, 0, 255, 128),
            3: (255, 255, 0, 128),
            4: (255, 0, 255, 128),
            5: (0, 255, 255, 128),
            6: (128, 0, 128, 128),
            7: (255, 165, 0, 128),
        }
        self.__setup_dataset_dirs()

    def __setup_dataset_dirs(self) -> None:
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")

    def save_image(
        self, image: Image.Image, image_filename: str, set_name: str
    ) -> None:
        image_name = os.path.splitext(os.path.basename(image_filename))[0]
        image_filename = f"{image_name}_{set_name}.jpg"
        image_path = os.path.join(self.check_images_dir, image_filename)
        image.save(image_path)

    def draw_annotations(
        self, image: Image.Image, kps_obj: AbsoluteKeypointsObject
    ) -> Image.Image:
        if not kps_obj.keypoints:
            return image

        # Draw rectangle
        draw = ImageDraw.Draw(image, "RGBA")
        box = [kps_obj.bbox[0], kps_obj.bbox[2]]
        draw.rectangle(box, outline=self.colors[kps_obj.class_idx], width=10)

        # Draw keypoints
        for kp in kps_obj.keypoints:
            if kp.visible == 1:
                draw.circle(
                    (kp.x, kp.y), radius=15, fill=self.colors[kps_obj.class_idx]
                )

        return image

    def visualize(self, num_images: int = 10) -> None:
        for set_dir in (self.train_dir, self.val_dir):
            set_name = os.path.basename(set_dir)
            label_path = os.path.join(set_dir, "labels.json")
            labels_dict = FileProcessor.read_json(label_path)

            # Get images listdir
            images_listdir = list(labels_dict.keys())
            random.shuffle(images_listdir)
            images_listdir = images_listdir[:num_images]

            for image_filename in tqdm(
                images_listdir, desc=f"Visualizing {set_name} images"
            ):
                # Load image
                image_path = os.path.join(set_dir, image_filename)
                image = Image.open(image_path)
                img_width, img_height = image.size

                # Create keypoints object
                abs_kps_obj = KeypointAugmenter.create_abs_kps_obj_from_label(
                    labels_dict[image_filename], clip=False
                )

                # Draw and save image
                drawn_image = self.draw_annotations(image, abs_kps_obj)
                self.save_image(drawn_image, image_filename, set_name)
