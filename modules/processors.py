from typing import List, Tuple

import os
import random
from PIL import Image

import numpy as np
import cv2

import pypdfium2 as pdfium
import doxapy


class ImageProcessor:
    def __init__(self) -> None:
        # Scan
        self.scan_types = ["bw", "gray", "color"]

        # Blue remove range
        self.lower_blue = np.array([70, 30, 30])
        self.upper_blue = np.array([130, 255, 255])

        # Binarization
        self.bin_params = {"window": 30, "k": 0.16}

    def remove_color(self,
                     img: np.ndarray) -> np.ndarray:
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

    def illuminate_image(self,
                         img: np.array,
                         alpha: float = 1.1,
                         beta=1) -> None:

        # Change luminance
        img = cv2.convertScaleAbs(img,
                                  alpha=alpha,
                                  beta=beta)

        return img

    def binarize_image(self,
                       img: np.ndarray) -> np.ndarray:
        # Convert image to gray
        if len(img.shape) != 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Create empty binary image
        bin_img = np.empty(gray.shape, gray.dtype)

        # Convert the image to binary
        wan = doxapy.Binarization(doxapy.Binarization.Algorithms.WAN)
        wan.initialize(gray)
        wan.to_binary(bin_img, self.bin_params)

        # Convert image back to 3d
        img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

        return img

    def resize_image(self,
                     img: np.ndarray,
                     max_height: int = 2000) -> np.ndarray:

        height, width = img.shape[:2]

        # Check if the height is greater than the specified max height
        if height > max_height:
            # Calculate the aspect ratio
            aspect_ratio = width / height
            # Calculate the new dimensions
            new_height = max_height
            new_width = int(new_height * aspect_ratio)

            # Resize the image
            img = cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return img

    def process(self,
                image: Image.Image,
                scan_type: str,
                resize: bool = False,
                remove_ink: bool = False,
                illuminate: bool = False,
                binarize: bool = False) -> Image.Image:
        # Check scan type
        assert scan_type in self.scan_types, f"Scan type should be in one of {str(self.scan_types)}"

        # Convert image to array
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Resize image
        if resize is True:
            img = self.resize_image(img)

        # Remove ink
        if remove_ink is True:
            img = self.remove_color(img)

        if illuminate is True:
            img = self.illuminate_image(img)

        # Binarize image
        if binarize is True and scan_type != "bw":
            img = self.binarize_image(img)

        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class PDFHandler:
    @staticmethod
    def create_pdf(images: List[Image.Image], output_path: str) -> None:
        pdf = pdfium.PdfDocument.new()

        for image in images:
            bitmap = pdfium.PdfBitmap.from_pil(image)
            pdf_image = pdfium.PdfImage.new(pdf)
            pdf_image.set_bitmap(bitmap)

            width, height = pdf_image.get_size()
            matrix = pdfium.PdfMatrix().scale(width, height)
            pdf_image.set_matrix(matrix)

            page = pdf.new_page(width, height)
            page.insert_obj(pdf_image)
            page.gen_content()

            bitmap.close()

        pdf.save(output_path, version=17)

    def get_page_image(self,
                       page: pdfium.PdfPage,
                       scale: int = 3) -> Image.Image:
        bitmap = page.render(scale=scale, rotation=0)
        image = bitmap.to_pil()
        return image if image.mode == 'RGB' else image.convert('RGB')

    def get_random_image(self,
                         pdf_listdir: List[str],
                         pdf_dir: str) -> Tuple[str, int, Image.Image]:
        # Take random pdf
        rand_pdf_name = random.choice(pdf_listdir)
        rand_pdf_path = os.path.join(pdf_dir, rand_pdf_name)
        rand_pdf_obj = pdfium.PdfDocument(rand_pdf_path)

        # Take random pdf page and image
        rand_page_idx = random.randint(0, len(rand_pdf_obj) - 1)
        rand_page = rand_pdf_obj[rand_page_idx]

        # Get random image and name
        rand_image = self.get_page_image(page=rand_page)
        rand_image_name = os.path.splitext(rand_pdf_name)[0] + ".jpg"

        # Close pdf file-object
        rand_pdf_obj.close()

        return rand_image, rand_image_name, rand_page_idx


class ImageHandler:
    @staticmethod
    def crop_image(image: Image.Image,
                   points: List[float],
                   offset: float = 0.025) -> Image.Image:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]

        pts = np.array([(int(x * width), int(y * height)) for x, y in points])
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        img = img[y:y+h, x:x+w].copy()

        pts = pts - pts.min(axis=0)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        result = cv2.bitwise_and(img, img, mask=mask)
        bg = np.ones_like(img, np.uint8)*255
        cv2.bitwise_not(bg, bg, mask=mask)
        result = bg + result

        border = int(height*offset)
        result = cv2.copyMakeBorder(result, border, border, border, border,
                                    cv2.BORDER_CONSTANT, value=[255, 255, 255])

        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    def get_random_image(self,
                         images_listdir: List[str],
                         images_dir: str) -> Tuple[Image.Image, str]:
        rand_image_name = random.choice(images_listdir)
        rand_image_path = os.path.join(images_dir, rand_image_name)
        rand_image = Image.open(rand_image_path)

        return rand_image, rand_image_name
