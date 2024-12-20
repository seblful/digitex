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

    @staticmethod
    def _draw_image(image: Image,
                    polygons: list[tuple[int, int]]):
        draw = ImageDraw.Draw(image, 'RGBA')
        for polygon in polygons:
            draw.polygon(polygon,
                         fill=((0, 255, 0, 128)),
                         outline="red")

        return image

    @staticmethod
    def _read_ann_file(ann_path: str) -> list[str]:
        with open(ann_path, 'r', encoding="utf-8") as ann_file:
            anns = ann_file.readlines()

        return anns

    @staticmethod
    def _get_polygons(anns: list[str]) -> list[tuple[int, int]]:
        polygons = []
        for ann in anns:
            polygon = ann.rstrip().split(',')[:-1]
            polygon = [pol for pol in polygon if pol.isdigit()]
            polygon = list(map(int, polygon))
            polygon = list(zip(polygon[::2], polygon[1::2]))
            polygons.append(polygon)

        return polygons

    def __extract_polygons(self,
                           ann_path: str) -> list[tuple[int, int]]:
        anns = self._read_ann_file(ann_path)
        polygons = self._get_polygons(anns)

        return polygons

    def __save_image(self,
                     image: Image,
                     image_name: str,
                     images_dir: str) -> None:
        image_name = os.path.splitext(image_name)[0]
        set_dir = [
            "train", "test"]["training" not in os.path.basename(images_dir)]
        save_image_name = f"{image_name}_{set_dir}.jpg"
        save_path = os.path.join(self.check_images_dir, save_image_name)
        image.save(save_path)

    def visualize(self, num_images: int = 10) -> None:
        print("Visualizing dataset images...")
        # Iterating through each dict and name of dataset that corresponds dict
        for set_dirs in [self.train_dirs, self.test_dirs]:
            images_dir, ann_dir = set_dirs
            data_dict = self.__create_data_dict(images_dir=images_dir,
                                                ann_dir=ann_dir,
                                                num_images=num_images)

            # Iterating through each image and annotation from directory
            for image_name, ann_name in data_dict.items():
                image_path = os.path.join(images_dir, image_name)
                ann_path = os.path.join(ann_dir, ann_name)

                image = Image.open(image_path)
                polygons = self.__extract_polygons(ann_path)
                drawn_image = self._draw_image(image=image,
                                               polygons=polygons)
                self.__save_image(image=drawn_image,
                                  images_dir=images_dir,
                                  image_name=image_name)
