import os
import random
from PIL import Image

import torch
from torchvision.io import read_image
import torchvision.transforms as T

from tqdm import tqdm

from digitex.core.processors.file import FileProcessor

from .annotation import AnnotationCreator
from .augmenter import KeypointAugmenter


class DatasetCreator:
    def __init__(
        self,
        raw_dir,
        dataset_dir,
        image_size: tuple[int, int],
        max_keypoints: int,
        train_split=0.8,
    ) -> None:
        self.raw_dir = raw_dir
        self.raw_images_dir = os.path.join(raw_dir, "images")
        self.data_json_path = os.path.join(raw_dir, "data.json")
        self.anns_json_path = os.path.join(raw_dir, "anns.json")

        self.dataset_dir = dataset_dir
        self._setup_dataset_dirs()

        self.image_size = image_size

        # Data split
        self.train_split = train_split
        self.val_split = 1 - self.train_split

        # Image resizing
        self._resize_image = T.Resize(self.image_size, antialias=True)

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

    def _resize_coords(
        self, vis_coords, orig_image_size: tuple[int, int]
    ) -> list[tuple[int, int]]:
        if not vis_coords:
            return []

        scale_x = self.image_size[1] / orig_image_size[1]
        scale_y = self.image_size[0] / orig_image_size[0]
        return [(x * scale_x, y * scale_y) for x, y in vis_coords]

    def _transform_and_save_image(
        self, set_dir: str, image_filename: str
    ) -> tuple[int, int]:
        img = read_image(os.path.join(self.raw_images_dir, image_filename))
        img_height, img_width = img.shape[1], img.shape[2]
        img = self._resize_image(img)
        img = img.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        image = Image.fromarray(img)
        image.save(os.path.join(set_dir, "images", image_filename))

        return img_height, img_width

    def _transform_label(
        self,
        label: list[tuple[int, int]],
        orig_img_size: tuple[int, int],
    ) -> None:
        abs_kps_obj = KeypointAugmenter.create_abs_kps_obj_from_label(label, clip=False)
        abs_kps_obj.resize_keypoints(
            orig_img_size[1], orig_img_size[0], self.image_size[1], self.image_size[0]
        )
        transf_label = abs_kps_obj.get_label()

        return transf_label

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
                img_size = self._transform_and_save_image(set_dir, image_filename)

                # Label processing
                label = total_labels_dict[image_filename]
                transf_label = self._transform_label(label, img_size)

                set_labels_dict[image_filename] = transf_label

            # Write labels
            label_path = os.path.join(set_dir, "labels.json")
            FileProcessor.write_json(set_labels_dict, label_path)

    def create_dataset(self) -> None:
        self.anns_creator.create_annotations()
        self._partitionate_data()
