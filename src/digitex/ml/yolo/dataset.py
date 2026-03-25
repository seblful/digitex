import logging
import random
import shutil
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class DatasetCreator:
    def __init__(
        self,
        raw_dir: str | Path,
        dataset_dir: str | Path,
        train_split: float = 0.8,
    ) -> None:

        self.raw_dir = Path(raw_dir)
        self.dataset_dir = Path(dataset_dir)

        self.__images_path: Path | None = None
        self.__labels_path: Path | None = None
        self.__train_dir: Path | None = None
        self.__val_dir: Path | None = None
        self.__test_dir: Path | None = None
        self.__images_labels_dict: dict[str, str] | None = None
        self.__classes_path: Path | None = None
        self.__data_yaml_path: Path | None = None

        self.anns_types = ["polygon"]

        self.__id2label: dict[int, str] | None = None
        self.__label2id: dict[str, int] | None = None

        self.train_split = train_split
        self.val_split = 0.6 * (1 - self.train_split)
        self.test_split = 1 - self.train_split - self.val_split

    @property
    def images_path(self) -> Path:
        if self.__images_path is None:
            self.__images_path = self.raw_dir / "images"

        return self.__images_path

    @property
    def labels_path(self) -> Path:
        if self.__labels_path is None:
            self.__labels_path = self.raw_dir / "labels"

        return self.__labels_path

    @property
    def train_dir(self) -> Path:
        if self.__train_dir is None:
            train_dir = self.dataset_dir / "train"
            train_dir.mkdir(parents=True, exist_ok=True)
            self.__train_dir = train_dir

        return self.__train_dir

    @property
    def val_dir(self) -> Path:
        if self.__val_dir is None:
            val_dir = self.dataset_dir / "val"
            val_dir.mkdir(parents=True, exist_ok=True)
            self.__val_dir = val_dir

        return self.__val_dir

    @property
    def test_dir(self) -> Path:
        if self.__test_dir is None:
            test_dir = self.dataset_dir / "test"
            test_dir.mkdir(parents=True, exist_ok=True)
            self.__test_dir = test_dir

        return self.__test_dir

    @property
    def data_yaml_path(self) -> Path:
        if self.__data_yaml_path is None:
            self.__data_yaml_path = self.dataset_dir / "data.yaml"

        return self.__data_yaml_path

    @property
    def classes_path(self) -> Path:
        if self.__classes_path is None:
            self.__classes_path = self.raw_dir / "classes.txt"

        return self.__classes_path

    @property
    def id2label(self) -> dict[int, str]:
        if self.__id2label is None:
            classes = DatasetCreator.read_classes_file(self.classes_path)
            self.__id2label = {k: v for k, v in enumerate(classes)}

        return self.__id2label

    @property
    def label2id(self) -> dict[str, int]:
        if self.__label2id is None:
            self.__label2id = {v: k for k, v in self.id2label.items()}

        return self.__label2id

    @property
    def images_labels_dict(self) -> dict[str, str]:
        if self.__images_labels_dict is None:
            self.__images_labels_dict = self.__create_images_labels_dict()

        return self.__images_labels_dict

    def __create_images_labels_dict(self, shuffle: bool = True) -> dict[str, str]:
        images = self.images_path.iterdir()
        labels = self.labels_path.iterdir()
        label_names = [label.name for label in labels]

        images_labels = {}
        for image in images:
            label = image.stem + '.txt'

            if label in label_names:
                images_labels[image.name] = label
            else:
                images_labels[image.name] = ""

        if shuffle:
            keys = list(images_labels.keys())
            random.shuffle(keys)
            images_labels = {key: images_labels[key] for key in keys}

        return images_labels

    @staticmethod
    def read_classes_file(classes_path: str | Path) -> list[str]:
        classes_path = Path(classes_path)
        with classes_path.open('r') as classes_file:
            classes = [i.split('\n')[0] for i in classes_file.readlines()]

        return classes

    def write_data_yaml(self, anns_type: str) -> None:
        logger.info(f"Available classes: {list(self.id2label.values())}")

        data = {
            'path': str(self.dataset_dir.resolve()),
            'train': "train",
            'val': "val",
            'test': "test",
            'names': self.id2label,
        }

        with self.data_yaml_path.open('w', encoding="utf-8") as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    @staticmethod
    def copy_files_from_dict(
        key: str,
        value: str,
        images_path: Path,
        labels_path: Path,
        copy_to: Path,
    ) -> None:
        shutil.copyfile(images_path / key, copy_to / key)
        if value:
            shutil.copyfile(labels_path / value, copy_to / value)

    def partition_data(self) -> None:
        data = self.images_labels_dict

        num_train = int(len(data) * self.train_split)
        num_val = int(len(data) * self.val_split)
        num_test = int(len(data) * self.test_split)

        train_data = {key: data[key] for key in list(data.keys())[:num_train]}
        val_data = {
            key: data[key]
            for key in list(data.keys())[num_train : num_train + num_val]
        }
        test_data = {
            key: data[key]
            for key in list(data.keys())[
                num_train + num_val : num_train + num_val + num_test
            ]
        }

        for key, value in train_data.items():
            DatasetCreator.copy_files_from_dict(
                key=key,
                value=value,
                images_path=self.images_path,
                labels_path=self.labels_path,
                copy_to=self.train_dir,
            )

        for key, value in val_data.items():
            DatasetCreator.copy_files_from_dict(
                key=key,
                value=value,
                images_path=self.images_path,
                labels_path=self.labels_path,
                copy_to=self.val_dir,
            )

        for key, value in test_data.items():
            DatasetCreator.copy_files_from_dict(
                key=key,
                value=value,
                images_path=self.images_path,
                labels_path=self.labels_path,
                copy_to=self.test_dir,
            )

    def create(self, anns_type: str) -> None:
        if anns_type not in self.anns_types:
            raise ValueError(f"anns_type must be one of {self.anns_types}")

        logger.info("Dataset is creating...")
        self.partition_data()
        logger.info("Train, validation, test dataset has created.")
        self.write_data_yaml(anns_type)
        logger.info("data.yaml file has created.")
