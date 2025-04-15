import os
from .base import BaseDataCreator


class PageDataCreator(BaseDataCreator):
    def extract_pages(
        self, raw_dir: str, train_dir: str, scan_type: str, num_images: int
    ) -> None:
        pdf_listdir = [pdf for pdf in os.listdir(raw_dir) if pdf.endswith("pdf")]
        num_saved = 0

        while num_images != num_saved:
            rand_image, rand_image_name, rand_page_idx = (
                self.pdf_handler.get_random_image(
                    pdf_listdir=pdf_listdir, pdf_dir=raw_dir
                )
            )
            rand_image = self.img_processor.process(
                image=rand_image, scan_type=scan_type
            )
            num_saved = self._save_image(
                rand_page_idx,
                output_dir=train_dir,
                image=rand_image,
                image_name=rand_image_name,
                num_saved=num_saved,
                num_images=num_images,
            )
