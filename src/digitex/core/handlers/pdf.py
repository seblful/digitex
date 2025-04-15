import os
import random
from PIL import Image

import pypdfium2 as pdfium


class PDFHandler:
    @staticmethod
    def create_pdf(images: list[Image.Image], output_path: str) -> None:
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

    @staticmethod
    def open_pdf(pdf_path: str) -> pdfium.PdfDocument[str]:
        pdf_obj = pdfium.PdfDocument(pdf_path)

        return pdf_obj

    def get_page_image(self, page: pdfium.PdfPage, dpi: int = 96) -> Image.Image:
        scale = dpi / 72.0
        bitmap = page.render(scale=scale, rotation=0)
        image = bitmap.to_pil()

        image = image if image.mode == "RGB" else image.convert("RGB")

        return image

    def get_random_image(
        self, pdf_listdir: list[str], pdf_dir: str
    ) -> tuple[str, int, Image.Image]:
        # Take random pdf
        rand_pdf_name = random.choice(pdf_listdir)
        rand_pdf_path = os.path.join(pdf_dir, rand_pdf_name)
        rand_pdf_obj = self.open_pdf(rand_pdf_path)

        # Take random pdf page and image
        rand_page_idx = random.randint(0, len(rand_pdf_obj) - 1)
        rand_page = rand_pdf_obj[rand_page_idx]

        # Get random image and name
        rand_image = self.get_page_image(page=rand_page)
        rand_image_name = os.path.splitext(rand_pdf_name)[0] + ".jpg"

        # Close pdf file-object
        rand_pdf_obj.close()

        return rand_image, rand_image_name, rand_page_idx
