import os
import random
from PIL import Image

import pypdfium2 as pdfium

from modules.processors import ImageProcessor


class DataCreator:
    def __init__(self,
                 input_dir: str,
                 train_dir: str) -> None:
        # Paths
        self.input_dir = input_dir
        self.train_dir = train_dir

        # Image processor
        self.image_processor = ImageProcessor()

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
                               num_images: int) -> None:
        # Pdf listdir
        self.pdf_listdir = [pdf for pdf in os.listdir(
            self.input_dir) if pdf.endswith('pdf')]
        # Counter for saved images
        num_saved_images = 0

        while num_images != num_saved_images:

            # Take random pdf
            rand_pdf = random.choice(self.pdf_listdir)
            rand_pdf_name = os.path.splitext(rand_pdf)[0]
            rand_pdf_path = os.path.join(self.input_dir, rand_pdf)
            rand_pdf_obj = pdfium.PdfDocument(rand_pdf_path)

            # Take random pdf page and image
            rand_page_ind = random.randint(0, len(rand_pdf_obj) - 1)
            rand_page = rand_pdf_obj[rand_page_ind]

            # Get random image and preprocess it
            rand_image = self.get_page_image(page=rand_page)
            rand_image = self.image_processor.clean_image(image=rand_image)
            rand_image_name = f"{rand_pdf_name}_{rand_page_ind}.jpg"
            rand_image_path = os.path.join(self.train_dir, rand_image_name)

            if not os.path.exists(rand_image_path):
                rand_image.save(rand_image_path, "JPEG")
                num_saved_images += 1
                print(f"It was saved {num_saved_images}/{num_images} images.")

            rand_image.close()
            rand_pdf_obj.close()
