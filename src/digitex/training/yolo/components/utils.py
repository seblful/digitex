import os
import random

from PIL import Image

import numpy as np


def get_random_image(images_dir,
                     images_listdir: list[str]) -> tuple[np.ndarray, str]:
    image_name = random.choice(images_listdir)
    image_path = os.path.join(images_dir, image_name)
    image = Image.open(image_path)

    return image_name, image


def get_random_img(images_dir,
                   images_listdir: list[str]) -> tuple[np.ndarray, str]:
    img_name, image = get_random_image(images_dir, images_listdir)
    img = np.array(image)

    return img_name, img
