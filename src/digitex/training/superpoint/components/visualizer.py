import os
import random
from PIL import Image, ImageDraw
import matplotlib.cm

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_image

from tqdm import tqdm

from digitex.core.processors.file import FileProcessor
from digitex.training.superpoint.components.annotation import AbsoluteKeypointsObject
from digitex.training.superpoint.components.augmenter import KeypointAugmenter


class BaseVisualizer:
    def __init__(self, dataset_dir: str, check_images_dir: str) -> None:
        self.dataset_dir = dataset_dir
        self.check_images_dir = check_images_dir
        self.__setup_dataset_dirs()

    def __setup_dataset_dirs(self) -> None:
        if not os.path.exists(self.check_images_dir):
            os.mkdir(self.check_images_dir)
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")

    def save_image(
        self, image: Image.Image, image_filename: str, set_name: str, postfix: str
    ) -> None:
        image_name = os.path.splitext(os.path.basename(image_filename))[0]
        image_filename = f"{image_name}_{postfix}_{set_name}.jpg"
        image_path = os.path.join(self.check_images_dir, image_filename)

        image.save(image_path)

    def visualize(self, num_images: int = 10) -> None:
        raise NotImplementedError("visualize() must be implemented in subclasses.")


class KeypointVisualizer(BaseVisualizer):
    def __init__(self, dataset_dir: str, check_images_dir: str) -> None:
        super().__init__(dataset_dir, check_images_dir)
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

    def draw_annotations(
        self, image: Image.Image, kps_obj: AbsoluteKeypointsObject
    ) -> Image.Image:
        if not kps_obj.keypoints:
            return image

        draw = ImageDraw.Draw(image, "RGBA")
        box = [kps_obj.bbox[0], kps_obj.bbox[2]]
        draw.rectangle(
            box, outline=self.colors.get(kps_obj.class_idx, (255, 0, 0, 128)), width=5
        )

        for kp in kps_obj.keypoints:
            if kp.visible == 1:
                draw.ellipse(
                    (kp.x - 5, kp.y - 5, kp.x + 5, kp.y + 5),
                    fill=self.colors.get(kps_obj.class_idx, (255, 0, 0, 128)),
                )
        return image

    def visualize(self, num_images: int = 10) -> None:
        for set_dir in (self.train_dir, self.val_dir):
            set_name = os.path.basename(set_dir)
            label_path = os.path.join(set_dir, "labels.json")
            labels_dict = FileProcessor.read_json(label_path)

            images_listdir = list(labels_dict.keys())
            random.shuffle(images_listdir)
            images_listdir = images_listdir[:num_images]

            for image_filename in tqdm(
                images_listdir, desc=f"Visualizing {set_name} keypoints"
            ):
                image_path = os.path.join(set_dir, "images", image_filename)
                image = Image.open(image_path)
                abs_kps_obj = KeypointAugmenter.create_abs_kps_obj_from_label(
                    labels_dict[image_filename], clip=False
                )
                drawn_image = self.draw_annotations(image, abs_kps_obj)
                self.save_image(drawn_image, image_filename, set_name, "keypoints")


class HeatmapsVisualizer(BaseVisualizer):
    def __init__(
        self,
        dataset_dir: str,
        check_images_dir: str,
        heatmaps_dir: str = "heatmaps",
        mask_dir: str = "masks",
    ) -> None:
        super().__init__(dataset_dir, check_images_dir)
        self.heatmaps_dir = heatmaps_dir
        self.mask_dir = mask_dir

    def _normalize_heatmap(self, heatmaps: torch.Tensor, img_size: tuple[int, int]):
        combined_heatmap = torch.max(heatmaps, dim=0)[0]  # (h, w)

        # Resize heatmap to image size
        heatmap_resized = (
            F.interpolate(
                combined_heatmap.unsqueeze(0).unsqueeze(0),
                size=img_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )  # (h, w)

        # Normalize heatmap to [0, 255]
        heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (
            np.ptp(heatmap_resized) + 1e-6
        )

        return heatmap_norm

    def visualize(self, num_images: int = 10) -> None:
        for set_dir in (self.train_dir, self.val_dir):
            set_name = os.path.basename(set_dir)
            images_dir = os.path.join(set_dir, "images")
            heatmaps_dir = os.path.join(set_dir, self.heatmaps_dir)

            images_listdir = os.listdir(images_dir)
            random.shuffle(images_listdir)
            images_listdir = images_listdir[:num_images]

            for image_filename in tqdm(
                images_listdir, desc=f"Visualizing {set_name} heatmaps"
            ):
                base = os.path.splitext(image_filename)[0]
                image_path = os.path.join(images_dir, image_filename)
                heatmap_path = os.path.join(heatmaps_dir, f"{base}.pt")
                # Load image
                img = read_image(image_path).float() / 255.0  # (C, H, W)
                img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C), float32

                # Load heatmap
                heatmaps = torch.load(heatmap_path)  # (K, h, w)
                heatmap_norm = self._normalize_heatmap(heatmaps, img.shape[0:2])

                # Apply colormap (jet)
                cmap = matplotlib.cm.get_cmap("jet")
                heatmap_color = cmap(heatmap_norm)[:, :, :3]  # (h, w, 3), RGB float
                heatmap_color = (heatmap_color * 255).astype(np.uint8)
                heatmap_img = Image.fromarray(heatmap_color)

                # Overlay heatmap on image
                img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(img_uint8)
                overlay = Image.blend(
                    image.convert("RGBA"), heatmap_img.convert("RGBA"), alpha=0.5
                )

                # Save overlay image
                self.save_image(
                    overlay.convert("RGB"), image_filename, set_name, "heatmaps"
                )
