from typing import Dict

import os
from PIL import Image

from modules.processors import ImageProcessor
from modules.handlers import PDFHandler, ImageHandler, LabelHandler
from modules.predictors.segmentation import YOLO_SegmentationPredictor


class DataCreator:
    def __init__(self) -> None:
        self.image_processor = ImageProcessor()
        self.pdf_handler = PDFHandler()
        self.image_handler = ImageHandler()
        self.label_handler = LabelHandler()

    @staticmethod
    def _read_classes(classes_path: str) -> Dict[int, str]:
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
            return {i: cl for i, cl in enumerate(classes)}

    def _save_image(self,
                    *args,
                    train_dir: str,
                    image: Image.Image,
                    image_name: str,
                    num_saved: int,
                    num_images: int) -> int:

        # Create image path
        image_stem = os.path.splitext(image_name)[0]
        str_ids = "_".join([str(i) for i in args])
        image_path = os.path.join(
            train_dir, image_stem) + "_" + str_ids + ".jpg"

        if not os.path.exists(image_path):
            image.save(image_path, "JPEG")
            num_saved += 1
            print(f"{num_saved}/{num_images} images was saved.")

            image.close()

        return num_saved

    def extract_pages(self,
                      raw_dir: str,
                      train_dir: str,
                      scan_type: str,
                      num_images: int) -> None:

        # Pdf listdir
        pdf_listdir = [pdf for pdf in os.listdir(
            raw_dir) if pdf.endswith('pdf')]

        # Counter for saved images
        num_saved = 0

        while num_images != num_saved:
            rand_image, rand_image_name, rand_page_idx = self.pdf_handler.get_random_image(pdf_listdir=pdf_listdir,
                                                                                           pdf_dir=raw_dir)

            # Process image
            rand_image = self.image_processor.process(image=rand_image,
                                                      scan_type=scan_type)

            # Save image
            num_saved = self._save_image(rand_page_idx,
                                         train_dir=train_dir,
                                         image=rand_image,
                                         image_name=rand_image_name,
                                         num_saved=num_saved,
                                         num_images=num_images)

    def extract_questions(self,
                          page_raw_dir: str,
                          train_dir: str,
                          num_images: int) -> None:
        # Paths
        images_dir = os.path.join(page_raw_dir, "images")
        labels_dir = os.path.join(page_raw_dir, "labels")
        classes_path = os.path.join(page_raw_dir, "classes.txt")

        # Classes
        classes_dict = DataCreator._read_classes(classes_path)

        # Images listdir
        images_listdir = os.listdir(images_dir)

        # Counter for saved images
        num_saved = 0

        while num_images != num_saved:
            # Extract random image, points
            rand_image, rand_image_name = self.image_handler.get_random_image(images_listdir=images_listdir,
                                                                              images_dir=images_dir)

            rand_points_idx, rand_points = self.label_handler.get_points(image_name=rand_image_name,
                                                                         labels_dir=labels_dir,
                                                                         classes_dict=classes_dict,
                                                                         target_classes=["question"])
            rand_points = self.label_handler.points_to_abs_polygon(points=rand_points,
                                                                   image_width=rand_image.width,
                                                                   image_height=rand_image.height)

            # Crop image and add borders
            rand_image = self.image_handler.crop_image(image=rand_image,
                                                       points=rand_points)

            # Save image
            num_saved = self._save_image(rand_points_idx,
                                         train_dir=train_dir,
                                         image=rand_image,
                                         image_name=rand_image_name,
                                         num_saved=num_saved,
                                         num_images=num_images)



    def extract_parts(self,
                      question_raw_dir: str,
                      train_dir: str,
                      num_images: int,
                      target_classes: list[str] = ["answer", "number", "option", "question", "spec"]) -> None:
        # Paths
        images_dir = os.path.join(question_raw_dir, "images")
        labels_dir = os.path.join(question_raw_dir, "labels")
        classes_path = os.path.join(question_raw_dir, "classes.txt")

        # Classes
        classes_dict = DataCreator._read_classes(classes_path)

        # Images listdir
        images_listdir = os.listdir(images_dir)

        # Counter for saved images
        num_saved = 0

        while num_images != num_saved:
            # Extract random image, points
            rand_image, rand_image_name = self.image_handler.get_random_image(images_listdir=images_listdir,
                                                                              images_dir=images_dir)

            rand_points_idx, rand_points = self.label_handler.get_points(image_name=rand_image_name,
                                                                         labels_dir=labels_dir,
                                                                         classes_dict=classes_dict,
                                                                         target_classes=target_classes)
            if not rand_points:
                continue

            rand_points = self.label_handler.points_to_abs_polygon(points=rand_points,
                                                                   image_width=rand_image.width,
                                                                   image_height=rand_image.height)

            # Crop image
            rand_image = self.image_handler.crop_image(image=rand_image,
                                                       points=rand_points,
                                                       offset=0.0)

            # Save image
            num_saved = self._save_image(rand_points_idx,
                                         train_dir=train_dir,
                                         image=rand_image,
                                         image_name=rand_image_name,
                                         num_saved=num_saved,
                                         num_images=num_images)



    def extract_words(self,
                      parts_raw_dir: str,
                      train_dir: str,
                      num_images: int) -> None:
        # Paths
        images_dir = os.path.join(parts_raw_dir, "images")
        labels_dir = os.path.join(parts_raw_dir, "labels")
        classes_path = os.path.join(parts_raw_dir, "classes.txt")

        # Classes
        classes_dict = DataCreator._read_classes(classes_path)
        target_classes = ["text"]

        # Images listdir
        images_listdir = os.listdir(images_dir)

        # Counter for saved images
        num_saved = 0

        while num_images != num_saved:
            # Extract random image, points
            rand_image, rand_image_name = self.image_handler.get_random_image(images_listdir=images_listdir,
                                                                              images_dir=images_dir)

            rand_points_idx, rand_points = self.label_handler.get_points(image_name=rand_image_name,
                                                                         labels_dir=labels_dir,
                                                                         classes_dict=classes_dict,
                                                                         target_classes=target_classes)
            rand_points = self.label_handler.points_to_abs_polygon(points=rand_points,
                                                                   image_width=rand_image.width,
                                                                   image_height=rand_image.height)

            # Crop image
            rand_image = self.image_handler.crop_image(image=rand_image,
                                                       points=rand_points,
                                                       offset=0.0)

            # Save image
            num_saved = self._save_image(rand_points_idx,
                                         train_dir=train_dir,
                                         image=rand_image,
                                         image_name=rand_image_name,
                                         num_saved=num_saved,
                                         num_images=num_images)


