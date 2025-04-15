import os
from .base import BaseDataCreator


class PageDataCreator(BaseDataCreator):
    def extract(self, pdf_dir: str, train_dir: str, num_images: int) -> None:
        pdf_listdir = [pdf for pdf in os.listdir(pdf_dir) if pdf.endswith("pdf")]
        num_saved = 0

        while num_images != num_saved:
            rand_image, rand_image_name, rand_page_idx = self._get_pdf_random_image(
                pdf_listdir=pdf_listdir, pdf_dir=pdf_dir
            )
            rand_image = self._process_image(image=rand_image)
            num_saved = self._save_image(
                rand_page_idx,
                output_dir=train_dir,
                image=rand_image,
                image_name=rand_image_name,
                num_saved=num_saved,
                num_images=num_images,
            )
