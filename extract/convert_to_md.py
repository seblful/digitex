import os
from PIL import Image

import numpy as np
import cv2
import pypdfium2 as pdfium

from marker.models import load_all_models
from marker.convert import convert_single_pdf
from marker.output import save_markdown


# For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]


# Paths
HOME = os.getcwd()
DATA_DIR = os.path.join(HOME, "data")
INPUT_DIR = os.path.join(DATA_DIR, "inputs")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")

LANGS = ["ru", "en"]


def remove_color(img: np.array,
                 lower_blue=np.array([115, 150, 70]),
                 upper_blue=np.array([130, 255, 255])) -> None:
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Dilate the mask to cover entire pen marks
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Inpaint the masked region
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return img


def clean_image(image: Image.Image,
                remove_ink: float = True,
                binarize: float = False,
                blur: float = False) -> Image.Image:
    # Convert image to array
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Adjust contrast and brightness
    img = cv2.convertScaleAbs(img,
                              alpha=3,
                              beta=15)

    # Binarize image
    if binarize is True:
        _, img = cv2.threshold(
            img, 150, 255, cv2.THRESH_BINARY)

    # Blur
    if blur is True:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # Remove ink
    if remove_ink is True:
        img = remove_color(img=img)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def preprocess_pdf(pdf_path: str) -> None:
    # Open pdf
    pdf = pdfium.PdfDocument(pdf_path)

    # Iterate
    for page in pdf:
        bitmap = page.render(scale=3,
                             rotation=0)
        image = bitmap.to_pil()
        # image.show()
        # Clean image
        image = clean_image(image=image,
                            remove_ink=True,
                            binarize=False,
                            blur=False)
        image.show("final image")
        break


def main() -> None:
    # Load models
    model_lst = load_all_models()

    for pdf_name in os.listdir(INPUT_DIR):
        # Check if it was converted before
        if not os.path.exists(os.path.join(OUTPUT_DIR, os.path.splitext(pdf_name)[0])):
            full_pdf_path = os.path.join(INPUT_DIR, pdf_name)

            # Convert pdf
            full_text, images, out_meta = convert_single_pdf(
                full_pdf_path, model_lst, langs=LANGS,  ocr_all_pages=True)

            # Save markdown
            subfolder_dir = save_markdown(
                OUTPUT_DIR, pdf_name, full_text, images, out_meta)
            print(f"Saved markdown to the '{
                os.path.basename(subfolder_dir)}' folder.")

            break


if __name__ == "__main__":
    # main()
    preprocess_pdf(os.path.join(INPUT_DIR, "ЦТ Биология 2022.pdf"))
