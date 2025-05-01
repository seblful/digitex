import os
import random
from PIL import Image

from digitex.core.processors.img import ImgProcessor, ImgCropper
from digitex.core.processors.file import FileProcessor
from digitex.core.handlers.pdf import PDFHandler
from digitex.core.handlers.label import LabelHandler


class BaseDataCreator:
    def __init__(self) -> None:
        self.img_processor = ImgProcessor()
        self.img_cropper = ImgCropper()
        self.file_processor = FileProcessor()
        self.pdf_handler = PDFHandler()
        self.label_handler = LabelHandler()

    def _read_classes(self, classes_path: str) -> dict[int, str]:
        classes = self.file_processor.read_txt(classes_path)
        return {i: cl.strip() for i, cl in enumerate(classes)}

    def _get_listdir_random_image(
        self, images_listdir: list, images_dir: str
    ) -> tuple[Image.Image, str]:
        rand_image_name = random.choice(images_listdir)
        rand_image_path = os.path.join(images_dir, rand_image_name)
        rand_image = Image.open(rand_image_path)
        return rand_image, rand_image_name

    def _get_pdf_random_image(
        self, pdf_listdir: list, pdf_dir: str
    ) -> tuple[str, int, Image.Image]:
        return self.pdf_handler.get_random_image(
            pdf_listdir=pdf_listdir, pdf_dir=pdf_dir
        )

    def _process_image(self, image: Image.Image) -> Image.Image:
        img = self.img_processor.image2img(image)
        img = self.img_processor.remove_blue(img)
        image = self.img_processor.img2image(img)
        return image

    def _crop_image(
        self, image: Image.Image, polygon: list[tuple[int, int]], offset: float = 0.025
    ) -> Image.Image:
        img = self.img_processor.image2img(image)
        cropped_img = self.img_cropper.crop_img_by_polygon(img, polygon)
        bg_img = self.img_cropper.paste_img_on_background(cropped_img, offset)
        bg_image = self.img_processor.img2image(bg_img)
        return bg_image

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

    def _get_points(
        self, image_name: str, labels_dir: str, classes_dict: dict, target_classes: list
    ) -> tuple[int, list[float]]:
        return self.label_handler.get_points(
            image_name=image_name,
            labels_dir=labels_dir,
            classes_dict=classes_dict,
            target_classes=target_classes,
        )

    def _convert_points_to_polygon(
        self, points: list[float], image_width: int, image_height: int
    ) -> list[tuple[int, int]]:
        return self.label_handler.points_to_abs_polygon(
            points=points, image_width=image_width, image_height=image_height
        )

    def extract(self, *args):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def predict(self, *args):
        raise NotImplementedError("This method should be overridden by subclasses.")
