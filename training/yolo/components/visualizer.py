import os
import random

import numpy as np

import imgaug as ia
from skimage.transform import resize
import imageio

from .augmenter import Augmenter


class Visualizer:
    def __init__(self,
                 dataset_dir,
                 check_images_dir,
                 number_of_images=6) -> None:
        self.dataset_dir = dataset_dir
        self.train_dataset_dir = os.path.join(self.dataset_dir, 'train')
        self.val_dataset_dir = os.path.join(self.dataset_dir, 'val')
        self.test_dataset_dir = os.path.join(self.dataset_dir, 'test')

        self.check_images_dir = check_images_dir

        self.number_of_images = number_of_images

        self.__images_labels_train_dict = None
        self.__images_labels_val_dict = None
        self.__images_labels_test_dict = None

        self.image_labels_dicts = (self.images_labels_train_dict,
                                   self.images_labels_val_dict, self.images_labels_test_dict)
        self.dataset_names = ('train', 'val', 'test')

    @staticmethod
    def create_images_labels_dict(data_dir,
                                  number_of_images,
                                  shuffle=True) -> dict[str, str]:
        # List of all images and labels in directory
        images = [image for image in os.listdir(
            data_dir) if image.endswith('.jpg')]
        labels = [label for label in os.listdir(
            data_dir) if label.endswith('.txt')]

        # Create a dictionary to store the images and labels names
        images_labels = {}
        for image in images:
            label = image.rstrip('.jpg') + '.txt'

            if label in labels:
                images_labels[image] = label
            else:
                images_labels[image] = None

        if shuffle:
            # Shuffle the data
            keys = list(images_labels.keys())
            random.shuffle(keys)
            images_labels = {key: images_labels[key] for key in keys}

        # Slice dict to number of images
        images_labels = dict(list(images_labels.items())[:number_of_images])

        return images_labels

    @property
    def images_labels_train_dict(self) -> dict[str, str]:
        if self.__images_labels_train_dict is None:
            self.__images_labels_train_dict = Visualizer.create_images_labels_dict(data_dir=self.train_dataset_dir,
                                                                                   number_of_images=self.number_of_images,
                                                                                   shuffle=True)

        return self.__images_labels_train_dict

    @property
    def images_labels_val_dict(self) -> dict[str, str]:
        if self.__images_labels_val_dict is None:
            self.__images_labels_val_dict = Visualizer.create_images_labels_dict(data_dir=self.val_dataset_dir,
                                                                                 number_of_images=self.number_of_images,
                                                                                 shuffle=True)

        return self.__images_labels_val_dict

    @property
    def images_labels_test_dict(self) -> dict[str, str]:
        if self.__images_labels_test_dict is None:
            self.__images_labels_test_dict = Visualizer.create_images_labels_dict(data_dir=self.test_dataset_dir,
                                                                                  number_of_images=self.number_of_images,
                                                                                  shuffle=True)

        return self.__images_labels_test_dict

    @staticmethod
    def resize_images(images, output_shape=(1024, 1024)) -> np.ndarray:
        resized_images = np.zeros(
            (len(images), output_shape[0], output_shape[1], images[0].shape[2]), dtype=images[0].dtype)

        for i, image in enumerate(images):
            # Resize the image using the `resize` function from the `skimage.transform` module
            resized_image = resize(image,
                                   output_shape,
                                   mode='reflect',
                                   preserve_range=True,
                                   anti_aliasing=True,
                                   clip=True,
                                   cval=0.0,
                                   order=3)

            resized_images[i] = resized_image

        return resized_images

    @staticmethod
    def write_cells(cells,
                    name_of_dataset,
                    check_images_dir) -> None:
        # Resize images to the same shape
        resized_images = Visualizer.resize_images(cells, (1024, 1024))

        # Convert cells to a grid image and save
        check_images_name = f"check_{name_of_dataset}_images.jpg"
        check_images_save_path = os.path.join(
            check_images_dir, check_images_name)
        os.makedirs(check_images_dir, exist_ok=True)

        grid_image = ia.draw_grid(
            resized_images, cols=4)
        imageio.v3.imwrite(check_images_save_path, grid_image)

    @staticmethod
    def visualize_images_with_labels(split_dataset_dir,
                                     image_array,
                                     label_name) -> np.ndarray:
        label_path = os.path.join(split_dataset_dir, label_name)

        # Extract points from label file
        points_with_labels = Augmenter.parse_labels_file(label_path)
        # Convert original points to a Polygon objects and convert points
        augmented_polygons = Augmenter.extract_and_convert_polygons(points_with_labels=points_with_labels,
                                                                    image_array=image_array)

        # Create copy of original image
        labelled_image = np.copy(image_array)

        # Iterating through all polygons and draw polygon on image
        for pols in augmented_polygons:
            labelled_image = pols.draw_on_image(labelled_image)

        return labelled_image

    def visualize(self) -> None:
        print("Visualizing dataset images...")
        # Iterating through each dict and name of dataset that corresponds dict
        for image_labels_dict, name_of_dataset in zip(self.image_labels_dicts, self.dataset_names):
            # Creating list for storing images and vizualization
            cells = []
            # Name of split dataset path
            split_dataset_dir = os.path.join(
                self.dataset_dir, name_of_dataset)
            # Iterating through each image and label from directory
            for image_name, label_name in image_labels_dict.items():
                image_path = os.path.join(split_dataset_dir, image_name)
                image_array = Augmenter.load_image(image_path=image_path)

                # Process images and append processed images to cells
                if label_name is not None:
                    labelled_image = Visualizer.visualize_images_with_labels(split_dataset_dir=split_dataset_dir,
                                                                             image_array=image_array,
                                                                             label_name=label_name)
                    cells.append(labelled_image)
                else:
                    cells.append(image_array)

            # Write cells to one image
            Visualizer.write_cells(cells=cells,
                                   name_of_dataset=name_of_dataset,
                                   check_images_dir=self.check_images_dir)

            print(f"Visualization for {name_of_dataset} set has finished.")
