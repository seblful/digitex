import os
import random
from PIL import Image

import pypdfium2 as pdfium

from modules.processors import ImageProcessor


class DataCreator:
    def __init__(self) -> None:

        # Image processor
        self.image_processor = ImageProcessor()

    def create_pdf_from_images(self,
                               image_dir: str,
                               raw_dir: str) -> None:
        # Create new pdf object
        pdf = pdfium.PdfDocument.new()

        # Iterate through images
        for image_name in os.listdir(image_dir):
            # Load image
            image_path = os.path.join(image_dir, image_name)
            image = pdfium.PdfImage.new(pdf)
            image.load_jpeg(image_path)
            width, height = image.get_size()

            # Create, scale and set_matrix
            matrix = pdfium.PdfMatrix().scale(width, height)
            image.set_matrix(matrix)

            # Create page and insert image to it
            page = pdf.new_page(width, height)
            page.insert_obj(image)
            page.gen_content()

        # Save pdf
        images_dir_name = os.path.basename(image_dir)
        raw_dir_name = os.path.basename(raw_dir)
        pdf_name = images_dir_name + " " + raw_dir_name + ".pdf"
        pdf_path = os.path.join(raw_dir, pdf_name)
        pdf.save(pdf_path, version=17)

    def get_page_image(self,
                       page: pdfium.PdfPage,
                       scale: int = 3) -> Image.Image:
        # Get image from pdf
        bitmap = page.render(scale=scale,
                             rotation=0)
        image = bitmap.to_pil()

        # Check image mode and convert if not RGB
        image_mode = image.mode

        if image_mode != 'RGB':
            image = image.convert('RGB')

        return image

    def create_yolo_train_data(self,
                               raw_dir: str,
                               train_dir: str,
                               scan_type: str,
                               num_images: int) -> None:

        # Pdf listdir
        pdf_listdir = [pdf for pdf in os.listdir(
            raw_dir) if pdf.endswith('pdf')]
        # Counter for saved images
        num_saved_images = 0

        while num_images != num_saved_images:

            # Take random pdf
            rand_pdf = random.choice(pdf_listdir)
            rand_pdf_name = os.path.splitext(rand_pdf)[0]
            rand_pdf_path = os.path.join(raw_dir, rand_pdf)
            rand_pdf_obj = pdfium.PdfDocument(rand_pdf_path)

            # Take random pdf page and image
            rand_page_ind = random.randint(0, len(rand_pdf_obj) - 1)
            rand_page = rand_pdf_obj[rand_page_ind]

            # Get random image and preprocess it
            rand_image = self.get_page_image(page=rand_page)
            rand_image = self.image_processor.process(image=rand_image,
                                                      scan_type=scan_type)
            rand_image_name = f"{rand_pdf_name}_{rand_page_ind}.jpg"
            rand_image_path = os.path.join(train_dir, rand_image_name)

            if not os.path.exists(rand_image_path):
                rand_image.save(rand_image_path, "JPEG")
                num_saved_images += 1
                print(f"It was saved {num_saved_images}/{num_images} images.")

            rand_image.close()
            rand_pdf_obj.close()

    def create_lm3_train_data(self):
        pass
