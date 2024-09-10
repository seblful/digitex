import os
import shutil
import random


class DatasetCreator():
    def __init__(self,
                 raw_dir: str,
                 dataset_dir: str,
                 train_split: float = 0.8) -> None:
        # Paths
        self.raw_dir = raw_dir
        self.dataset_dir = dataset_dir

        self._masks_dir = None

        self.__train_dir = None
        self.__val_dir = None
        self.__test_dir = None

        # Input dirs
        self.json_path = os.path.join(raw_dir, 'data.json')
        self.classes_path = os.path.join(raw_dir, 'classes.txt')

        # Data split
        self.train_split = train_split
        self.val_split = 0.6 * (1 - self.train_split)
        self.test_split = 1 - self.train_split - self.val_split

        self.__id2label = None
        self.__label2id = None

        self.__images_labels_dict = None

    @property
    def masks_dir(self) -> str:
        if self.__masks_dir is None:
            masks_dir = os.path.join(self.raw_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            self.__masks_dir = masks_dir

        return self.__masks_dir

    @property
    def train_dir(self) -> str:
        if self.__train_dir is None:
            train_dir = os.path.join(self.dataset_dir, 'train')
            images_dir = os.path.join(train_dir, 'images')
            masks_dir = os.path.join(train_dir, 'masks')
            # Creating folder for train set
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

            self.__train_dir = train_dir

        return self.__train_dir

    @property
    def val_dir(self) -> str:
        if self.__val_dir is None:
            val_dir = os.path.join(self.dataset_dir, 'val')
            images_dir = os.path.join(val_dir, 'images')
            masks_dir = os.path.join(val_dir, 'masks')
            # Creating folder for val set
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

            self.__val_dir = val_dir

        return self.__val_dir

    @property
    def test_dir(self) -> str:
        if self.__test_dir is None:
            test_dir = os.path.join(self.dataset_dir, 'test')
            images_dir = os.path.join(test_dir, 'images')
            masks_dir = os.path.join(test_dir, 'masks')
            # Creating folder for test set
            os.makedirs(test_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

            self.__test_dir = test_dir

        return self.__test_dir

    @property
    def images_labels_dict(self) -> dict:
        if self.__images_labels_dict is None:
            self.__images_labels_dict = self.__create_images_labels_dict()

        return self.__images_labels_dict

    def __create_images_labels_dict(self, shuffle=True) -> dict:
        # List of all images and labels in directory
        # images = os.listdir(self.images_dir)
        labels = os.listdir(self.masks_dir)

        # Create a dictionary to store the images and labels names
        images_labels = {}
        for label in labels:
            image = os.path.splitext(label)[0] + '.jpg'

            images_labels[image] = label

        if shuffle:
            # Shuffle the data
            keys = list(images_labels.keys())
            random.shuffle(keys)
            images_labels = {key: images_labels[key] for key in keys}

        return images_labels

    @property
    def id2label(self) -> dict[int, str]:
        if self.__id2label is None:
            self.__id2label = self.__create_id2label()

        return self.__id2label

    @property
    def label2id(self) -> dict[str, int]:
        if self.__label2id is None:
            self.__label2id = {v: k for k, v in self.id2label.items()}

        return self.__label2id

    def __create_id2label(self) -> dict[int, str]:
        with open(self.classes_path, 'r') as file:
            # Set the names of the classes
            classes = [i.split('\n')[0] for i in file.readlines()]
            id2label = {k: v for k, v in enumerate(classes)}

        return id2label

    def process(self) -> None:
        self.train_dir
        self.val_dir
        self.test_dir
