import os
from PIL import Image

from data_creator_new import PDFHandler
from processors import ImageProcessor


def create_pdf_from_images(image_dir: str,
                           raw_dir: str,
                           process: bool = False) -> None:
    # Sort image listdir
    def num_key(x) -> int: return int(x.split("_")[-1].split(".")[0])
    image_listdir = sorted(os.listdir(image_dir), key=num_key)

    # Iterate through images and preprocess
    images = []
    for image_name in image_listdir:
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)

        if process:
            image = ImageProcessor().process(image=image,
                                             scan_type="color")

        images.append(image)

    # Save pdf
    pdf_name = f"{os.path.basename(image_dir)} {
        os.path.basename(raw_dir)}.pdf"
    pdf_path = os.path.join(raw_dir, pdf_name)
    PDFHandler().create_pdf(images, pdf_path)
