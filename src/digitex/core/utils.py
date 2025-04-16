import os
from PIL import Image

import numpy as np
import cv2

from digitex.core.handlers.pdf import PDFHandler


def crop_image(
    image: Image.Image, points: list[float], offset: float = 0.025
) -> Image.Image:
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = img.shape[:2]

    pts = np.array(points)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    img = img[y : y + h, x : x + w].copy()

    pts = pts - pts.min(axis=0)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    result = cv2.bitwise_and(img, img, mask=mask)
    bg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    result = bg + result

    border = int(height * offset)
    result = cv2.copyMakeBorder(
        result,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


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
