import os
import random

import numpy as np
from skimage.transform import resize
import imageio

from .augmenter import self


class Visualizer:
    def __init__(self,
                 dataset_dir,
                 check_images_dir,
                 number_of_images=6) -> None:
        self.dataset_dir = dataset_dir

        self.check_images_dir = check_images_dir

        self.number_of_images = number_of_images

        self.dataset_names = ('train', 'val', 'test')

    def visualize(self) -> None:
        print("Visualizing dataset images...")
