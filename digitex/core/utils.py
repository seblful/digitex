import os
from PIL import Image

from digitex.core.img import ImgProcessor
from digitex.core.handlers import PDFHandler


def process_image(image: Image) -> Image:
    img = ImgProcessor.image2img(image=image)
    img = ImgProcessor.remove_blue(img)
    image = ImgProcessor.img2image(img=img)

    # img = ImgProcessor.resize_image(img=img, target_width=1000, target_height=1000)

    return image


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
