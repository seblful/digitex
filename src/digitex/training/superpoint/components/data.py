import os
import shutil
import random

from tqdm import tqdm

from digitex.core.processors.file import FileProcessor
from digitex.training.superpoint.components.annotation import AnnotationCreator


class DatasetCreator:
    def __init__(
        self, raw_dir, dataset_dir, num_keypoints=None, train_split=0.8
    ) -> None:
        self.raw_dir = raw_dir
        self.raw_images_dir = os.path.join(raw_dir, "images")
        self.data_json_path = os.path.join(raw_dir, "data.json")
        self.anns_json_path = os.path.join(raw_dir, "anns.json")

        self.dataset_dir = dataset_dir
        self._setup_dataset_dirs()

        self.num_keypoints = num_keypoints

        # Data split
        self.train_split = train_split
        self.val_split = 1 - self.train_split

        self.anns_creator = AnnotationCreator(
            data_json_path=self.data_json_path,
            anns_json_path=self.anns_json_path,
            num_keypoints=num_keypoints,
        )

    def _setup_dataset_dirs(self) -> None:
        os.mkdir(self.dataset_dir)

        # Train dirs
        self.train_dir = os.path.join(self.dataset_dir, "train")
        train_images_dir = os.path.join(self.train_dir, "images")
        os.makedirs(train_images_dir)

        # Val dirs
        self.val_dir = os.path.join(self.dataset_dir, "val")
        val_images_dir = os.path.join(self.val_dir, "images")
        os.makedirs(val_images_dir)

    def _copy_data(self, listdir: list[str], set_dir: str, anns_dict: dict) -> None:
        labels_dict = {}
        for image_name in tqdm(
            listdir, desc=f"Partitioning {os.path.basename(set_dir)} data"
        ):
            # Copy image
            shutil.copyfile(
                os.path.join(self.raw_images_dir, image_name),
                os.path.join(set_dir, "images", image_name),
            )
            # Prepare annotation string
            path = os.path.join("images", image_name)
            ann = anns_dict[image_name]
            labels_dict[path] = ann

        label_path = os.path.join(set_dir, "labels.json")
        FileProcessor.write_json(labels_dict, label_path)

    def _partitionate_data(self) -> None:
        # Images listdir and shuffle
        images_listdir = os.listdir(self.raw_images_dir)
        random.shuffle(images_listdir)

        # Create train and validation listdirs
        num_train = int(len(images_listdir) * self.train_split)
        num_val = int(len(images_listdir) * self.val_split)
        train_listdir = images_listdir[:num_train]
        val_listdir = images_listdir[num_train : num_train + num_val]

        # Load anns dict
        anns_dict = FileProcessor.read_json(json_path=self.anns_json_path)

        # Copy the images to folders and create annotation file
        for listdir, set_dir in zip(
            (train_listdir, val_listdir), (self.train_dir, self.val_dir)
        ):
            self._copy_data(listdir=listdir, set_dir=set_dir, anns_dict=anns_dict)

    def create_dataset(self) -> None:
        self.anns_creator.create_annotations()
        self._partitionate_data()
