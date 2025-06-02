import os
import random
from PIL import Image

import numpy as np

from digitex.core.handlers.pdf import PDFHandler


def create_pdf_from_images(image_dir: str, output_dir: str) -> None:
    # Sort image listdir
    def num_key(x) -> int:
        return int(x.split("_")[-1].split(".")[0])

    image_listdir = sorted(os.listdir(image_dir), key=num_key)

    # Iterate through images and preprocess
    images = []
    for image_name in image_listdir:
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)

        images.append(image)

    # Save pdf
    pdf_name = f"{os.path.basename(image_dir)} {os.path.basename(output_dir)}.pdf"
    pdf_path = os.path.join(output_dir, pdf_name)
    PDFHandler().create_pdf(images, pdf_path)


def get_random_image(images_dir, images_listdir: list[str]) -> tuple[np.ndarray, str]:
    image_name = random.choice(images_listdir)
    image_path = os.path.join(images_dir, image_name)
    image = Image.open(image_path)

    return image_name, image


def get_random_img(images_dir, images_listdir: list[str]) -> tuple[np.ndarray, str]:
    img_name, image = get_random_image(images_dir, images_listdir)
    img = np.array(image)

    return img_name, img
