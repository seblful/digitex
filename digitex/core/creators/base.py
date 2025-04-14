import os
from typing import Dict
from PIL import Image

from digitex.core.img import ImgProcessor
from digitex.core.file import FileProcessor
from handlers import PDFHandler, ImageHandler, LabelHandler


class BaseDataCreator:
    def __init__(self) -> None:
        self.img_processor = ImgProcessor()
        self.file_processor = FileProcessor()
        self.pdf_handler = PDFHandler()
        self.image_handler = ImageHandler()
        self.label_handler = LabelHandler()

    def _read_classes(self, classes_path: str) -> Dict[int, str]:
        classes = self.file_processor.read_txt(classes_path)
        return {i: cl.strip() for i, cl in enumerate(classes)}

    def _save_image(
        self,
        *args,
        output_dir: str,
        image: Image.Image,
        image_name: str,
        num_saved: int,
        num_images: int,
    ) -> int:
        image_stem = os.path.splitext(image_name)[0]
        str_ids = "_".join([str(i) for i in args])
        image_path = os.path.join(output_dir, image_stem) + "_" + str_ids + ".jpg"

        if not os.path.exists(image_path):
            image.save(image_path, "JPEG")
            num_saved += 1
            print(f"{num_saved}/{num_images} images was saved.")
            image.close()

        return num_saved
