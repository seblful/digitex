import os
import random
import shutil
from PIL import Image

import torch

from tqdm import tqdm

from digitex.core.processors.file import FileProcessor

from .annotation import AnnotationCreator
from .augmenter import KeypointAugmenter


class DatasetCreator:
    def __init__(
        self,
        raw_dir,
        dataset_dir,
        max_keypoints: int,
        train_split=0.8,
    ) -> None:
        self.raw_dir = raw_dir
        self.raw_images_dir = os.path.join(raw_dir, "images")
        self.data_json_path = os.path.join(raw_dir, "data.json")
        self.anns_json_path = os.path.join(raw_dir, "anns.json")

        self.dataset_dir = dataset_dir
        self._setup_dataset_dirs()

        # Data split
        self.train_split = train_split
        self.val_split = 1 - self.train_split

        self.anns_creator = AnnotationCreator(
            data_json_path=self.data_json_path,
            anns_json_path=self.anns_json_path,
            num_keypoints=max_keypoints,
        )

    def _setup_dataset_dirs(self) -> None:
        os.mkdir(self.dataset_dir)

        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")

        for set_dir in [self.train_dir, self.val_dir]:
            os.mkdir(set_dir)
            os.mkdir(os.path.join(set_dir, "images"))

    def _train_val_split(self) -> tuple[list[str], list[str]]:
        # Images listdir and shuffle
        images_listdir = os.listdir(self.raw_images_dir)
        random.shuffle(images_listdir)

        # Create train and validation listdirs
        num_train = int(len(images_listdir) * self.train_split)
        num_val = int(len(images_listdir) * self.val_split)
        train_listdir = images_listdir[:num_train]
        val_listdir = images_listdir[num_train : num_train + num_val]

        return train_listdir, val_listdir

    def _copy_image(self, set_dir: str, image_filename: str) -> None:
        # Copy image
        src_path = os.path.join(self.raw_images_dir, image_filename)
        dst_path = os.path.join(set_dir, "images", image_filename)

        # Copy the image file
        shutil.copy2(src_path, dst_path)

        return None

    def _partitionate_data(self) -> None:
        # Split listdir
        train_listdir, val_listdir = self._train_val_split()

        # Load anns dict
        total_labels_dict = FileProcessor.read_json(json_path=self.anns_json_path)

        # Copy the images to folders and create annotation file
        for listdir, set_dir in zip(
            (train_listdir, val_listdir), (self.train_dir, self.val_dir)
        ):
            set_labels_dict = {}
            for image_filename in tqdm(
                listdir, desc=f"Partitioning {os.path.basename(set_dir)} data"
            ):
                # Image processing
                self._copy_image(set_dir, image_filename)

                # Label processing
                label = total_labels_dict[image_filename]
                set_labels_dict[image_filename] = label

            # Write labels
            label_path = os.path.join(set_dir, "labels.json")
            FileProcessor.write_json(set_labels_dict, label_path)

    def create_dataset(self) -> None:
        self.anns_creator.create_annotations()
        self._partitionate_data()


class HeatmapsCreator:
    def __init__(
        self,
        dataset_dir: str,
        max_keypoints: int,
        heatmap_size: tuple[int, int] = (256, 256),
        heatmap_sigma: float = 2.0,
    ) -> None:
        self.dataset_dir = dataset_dir
        self._setup_dataset_dirs()

        self.max_keypoints = max_keypoints
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma

    def _setup_dataset_dirs(self) -> None:
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")

        for set_dir in [self.train_dir, self.val_dir]:
            os.mkdir(os.path.join(set_dir, "heatmaps"))
            os.mkdir(os.path.join(set_dir, "masks"))

    @staticmethod
    def generate_heatmaps(
        points: list[torch.Tensor],
        image_size: tuple[int, int],
        heatmap_size: tuple[int, int],
        max_keypoints: int,
        heatmap_sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not points:
            heatmaps = torch.zeros(max_keypoints, heatmap_size[0], heatmap_size[1])
            mask = torch.zeros(max_keypoints, dtype=torch.float32)
            return (heatmaps, mask)

        heatmaps = torch.zeros(max_keypoints, heatmap_size[0], heatmap_size[1])
        mask = torch.zeros(max_keypoints, dtype=torch.float32)

        # Scale factors for coordinate transformation from image space to heatmap space
        h_scale = heatmap_size[0] / image_size[0]
        w_scale = heatmap_size[1] / image_size[1]

        # Create grid for heatmap generation
        y_grid, x_grid = torch.meshgrid(
            torch.arange(heatmap_size[0], dtype=torch.float32),
            torch.arange(heatmap_size[1], dtype=torch.float32),
            indexing="ij",
        )

        for p_idx, p in enumerate(points[:max_keypoints]):
            if p is None or torch.isnan(p).any():
                continue

            mask[p_idx] = 1.0

            # Scale keypoint coordinates from image space to heatmap space
            x = p[0] * w_scale
            y = p[1] * h_scale

            # Create Gaussian heatmap
            dist_sq = (x_grid - x) ** 2 + (y_grid - y) ** 2
            exponent = dist_sq / (2 * heatmap_sigma**2)
            heatmap = torch.exp(-exponent)

            # Normalize to 0-1 range
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            heatmaps[p_idx] = heatmap

        return (heatmaps, mask)

    def _create_points(self, vis_coords: list[tuple[int, int]]) -> list[torch.Tensor]:
        if not vis_coords:
            return []

        points = [torch.tensor([p[0], p[1]], dtype=torch.float32) for p in vis_coords]

        if len(points) < self.max_keypoints:
            points += [torch.full([2], float("nan"))] * (
                self.max_keypoints - len(points)
            )
        else:
            points = points[: self.max_keypoints]

        return points

    def _get_image_size(self, set_dir: str, image_filename: str) -> tuple[int, int]:
        image_path = os.path.join(set_dir, "images", image_filename)
        image = Image.open(image_path)
        img_width, img_height = image.size
        image.close()
        return img_height, img_width

    def _transform_and_save_heatmaps(
        self,
        set_dir: str,
        label: list[tuple[int, int]],
        image_filename: str,
    ) -> None:
        # Get the actual image size
        image_size = self._get_image_size(set_dir, image_filename)

        abs_kps_obj = KeypointAugmenter.create_abs_kps_obj_from_label(label, clip=False)
        vis_coords = abs_kps_obj.get_vis_coords()
        points = self._create_points(vis_coords)
        heatmaps, mask = self.generate_heatmaps(
            points,
            image_size,
            self.heatmap_size,
            self.max_keypoints,
            self.heatmap_sigma,
        )

        name = os.path.splitext(image_filename)[0]
        torch.save(heatmaps, os.path.join(set_dir, "heatmaps", f"{name}.pt"))
        torch.save(mask, os.path.join(set_dir, "masks", f"{name}.pt"))

    def create_heatmaps(self) -> None:
        for set_dir in [self.train_dir, self.val_dir]:
            label_path = os.path.join(set_dir, "labels.json")
            labels_dict = FileProcessor.read_json(label_path)

            for image_filename, label in tqdm(
                labels_dict.items(),
                desc=f"Creating heatmaps for {os.path.basename(set_dir)}",
            ):
                # Label processing
                self._transform_and_save_heatmaps(set_dir, label, image_filename)
