import os
from PIL import Image

import numpy as np
import cv2
import pypdfium2 as pdfium

from marker.models import load_all_models
from marker.convert import convert_single_pdf
from marker.output import save_markdown


class ImageProcessor:
    def __init__(self,
                 alpha: int,
                 beta: int,
                 remove_ink: bool,
                 binarize: bool,
                 blur: bool) -> None:
        # Parameters
        self.alpha = alpha
        self.beta = beta
        self.remove_ink = remove_ink
        self.binarize = binarize
        self.blur = blur

        # Blue remove range
        self.lower_blue = np.array([115, 150, 70])
        self.upper_blue = np.array([130, 255, 255])

    def remove_color(self,
                     img: np.array) -> None:
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create a mask for blue color
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        # Dilate the mask to cover entire pen marks
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Inpaint the masked region
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        return img

    def clean_image(self,
                    image: Image.Image) -> Image.Image:
        # Convert image to array
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Adjust contrast and brightness
        img = cv2.convertScaleAbs(img,
                                  alpha=self.alpha,
                                  beta=self.beta)

        # Binarize image
        if self.binarize is True:
            _, img = cv2.threshold(
                img, 150, 255, cv2.THRESH_BINARY)

        # Blur
        if self.blur is True:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        # Remove ink
        if self.remove_ink is True:
            img = self.remove_color(img=img)

        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class PDFProcessor:
    def __init__(self,
                 raw_dir: str,
                 input_dir: str,
                 output_dir: str,
                 alpha: int = 2,
                 beta: int = 10,
                 remove_ink: bool = True,
                 binarize: bool = False,
                 blur: bool = False) -> None:
        # Paths
        self.raw_dir = raw_dir
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Image processor
        self.image_processor = ImageProcessor(alpha=alpha,
                                              beta=beta,
                                              remove_ink=remove_ink,
                                              binarize=binarize,
                                              blur=blur)

        # Model, langs
        self.langs = ["ru", "en"]

    @staticmethod
    def insert_image(pdf: pdfium.PdfDocument,
                     image: Image.Image,
                     width: float,
                     height: float) -> None:
        # Insert image to pdf image
        pdf_image = pdfium.PdfImage.new(pdf)
        bitmap = pdfium.PdfBitmap.from_pil(image)
        image.close()
        pdf_image.set_bitmap(bitmap)
        bitmap.close()

        # Set matrix
        pdf_image.set_matrix(
            pdfium.PdfMatrix().scale(width, height))

        # Create page and insert pdf image
        page = pdf.new_page(width, height)
        page.insert_obj(pdf_image)
        page.gen_content()

        # Close
        pdf_image.close()
        page.close()

    def preprocess_pdf(self) -> None:
        # Iterate through each pdf in raw dir
        for pdf_name in os.listdir(self.raw_dir):
            # Check if it was preprocessed before
            if not os.path.exists(os.path.join(self.input_dir, pdf_name)):
                print(f"Preprocessing '{pdf_name}'...")
                # Open pdf
                full_pdf_path = os.path.join(self.raw_dir, pdf_name)
                raw_pdf = pdfium.PdfDocument(full_pdf_path)
                version = raw_pdf.get_version()

                # Create new pdf
                input_pdf = pdfium.PdfDocument.new()

                # Iterate
                for page in raw_pdf:
                    # Get width and height of the page
                    width, height = page.get_size()

                    # Retrieve image
                    bitmap = page.render(scale=3,
                                         rotation=0)
                    image = bitmap.to_pil()

                    # Clean image
                    image = self.image_processor.clean_image(image=image)

                    PDFProcessor.insert_image(pdf=input_pdf,
                                              image=image,
                                              width=width,
                                              height=height)

                # Save pdf
                input_pdf.save(dest=os.path.join(self.input_dir, pdf_name),
                               version=version)

                # Close pdfs
                raw_pdf.close()
                input_pdf.close()

    def convert_to_md(self) -> None:
        # Iterate through each pdf in input dir
        for pdf_name in os.listdir(self.input_dir):
            # Check if it was converted before
            if not os.path.exists(os.path.join(self.output_dir, os.path.splitext(pdf_name)[0])):
                print(f"Converting '{pdf_name}'...")
                full_pdf_path = os.path.join(self.input_dir, pdf_name)

                # Load models
                model_lst = load_all_models()
                # Convert pdf
                full_text, images, out_meta = convert_single_pdf(full_pdf_path, model_lst,
                                                                 langs=self.langs,
                                                                 ocr_all_pages=True)

                # Save markdown
                subfolder_dir = save_markdown(
                    self.output_dir, pdf_name, full_text, images, out_meta)
                print(f"Saved markdown to the '{
                    os.path.basename(subfolder_dir)}' folder.")

    def process(self) -> None:
        self.preprocess_pdf()
        self.convert_to_md()
