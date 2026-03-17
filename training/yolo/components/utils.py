import random
from pathlib import Path

from PIL import Image

import numpy as np


def get_random_image(
    images_dir: str, images_listdir: list[str]
) -> tuple[np.ndarray, str]:
    image_name = random.choice(images_listdir)
    image_path = Path(images_dir) / image_name
    image = Image.open(str(image_path))

    return image_name, image


def get_random_img(
    images_dir: str, images_listdir: list[str]
) -> tuple[np.ndarray, str]:
    img_name, image = get_random_image(images_dir, images_listdir)
    img = np.array(image)

    return img_name, img
