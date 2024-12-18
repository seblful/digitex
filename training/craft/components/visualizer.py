import os
import random
from PIL import Image, ImageDraw


class Visualizer:
    def __init__(self,
                 dataset_dir,
                 check_images_dir) -> None:
        self.dataset_dir = dataset_dir
        self.__setup_dataset_dirs()

        self.check_images_dir = check_images_dir

    def __setup_dataset_dirs(self) -> None:
        # Create paths
        images_train_dir = os.path.join(
            self.dataset_dir, "ch4_training_images")
        ann_train_dir = os.path.join(
            self.dataset_dir, "ch4_training_localization_transcription_gt")
        images_test_dir = os.path.join(
            self.dataset_dir, "ch4_test_images")
        ann_test_dir = os.path.join(
            self.dataset_dir, "ch4_test_localization_transcription_gt")

        # Create list of dirs
        self.train_dirs = [images_train_dir, ann_train_dir]
        self.test_dirs = [images_test_dir, ann_test_dir]

    @staticmethod
    def __create_data_dict(images_dir,
                           ann_dir,
                           num_images=5,
                           shuffle=True) -> dict[str, str]:
        # List of all images and annotations in directory
        images = [image for image in os.listdir(images_dir)]
        anns = [ann for ann in os.listdir(ann_dir)]

        # Create a dictionary to store the images and annotations names
        data_dict = {}
        for image in images:
            ann = f"gt_{os.path.splitext(image)[0]}.txt"

            if ann in anns:
                data_dict[image] = ann
            else:
                data_dict[image] = None

        # Shuffle the data
        if shuffle:
            keys = list(data_dict.keys())
            random.shuffle(keys)
            data_dict = {key: data_dict[key] for key in keys}

        # Slice dict to number of images
        data_dict = dict(list(data_dict.items())[:num_images])

        return data_dict
