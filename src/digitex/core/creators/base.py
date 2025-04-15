import os
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

    def _read_classes(self, classes_path: str) -> dict[int, str]:
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

    def _get_random_image(self, images_listdir: list, images_dir: str):
        return self.image_handler.get_random_image(
            images_listdir=images_listdir, images_dir=images_dir
        )

    def get_pdf_random_image(self, pdf_listdir: list, pdf_dir: str):
        return self.pdf_handler.get_random_image(
            pdf_listdir=pdf_listdir, pdf_dir=pdf_dir
        )

    def get_listdir_random_image(self, images_listdir: list, images_dir: str):
        return self._get_random_image(
            images_listdir=images_listdir, images_dir=images_dir
        )

    def _process_image(self, image, scan_type: str):
        return self.img_processor.process(image=image, scan_type=scan_type)

    def _crop_image(self, image, points, offset: float = 0.0):
        return self.image_handler.crop_image(image=image, points=points, offset=offset)

    def _get_points(
        self, image_name: str, labels_dir: str, classes_dict: dict, target_classes: list
    ):
        return self.label_handler.get_points(
            image_name=image_name,
            labels_dir=labels_dir,
            classes_dict=classes_dict,
            target_classes=target_classes,
        )

    def _convert_points_to_polygon(self, points, image_width: int, image_height: int):
        return self.label_handler.points_to_abs_polygon(
            points=points, image_width=image_width, image_height=image_height
        )
