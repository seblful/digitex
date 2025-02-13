import os
import random

from PIL import Image

import numpy as np


def get_random_img(images_dir,
                   images_listdir: list[str]) -> tuple[np.ndarray, str]:
    img_name = random.choice(images_listdir)
    img_path = os.path.join(images_dir, img_name)
    image = Image.open(img_path)
    img = np.array(image)

    return img_name, img
