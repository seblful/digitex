from typing import Dict

import os
import random
from PIL import Image

# from transformers import PreTrainedModel, SegformerImageProcessor
# from surya.model.detection.model import load_model as load_text_det_model, load_processor as load_text_det_processor
# from surya.detection import batch_text_detection

from modules.processors import ImageProcessor
from modules.handlers import PDFHandler, ImageHandler, LabelHandler
from modules.predictors import YoloPredictor


class DataCreator:
    def __init__(self) -> None:
        self.image_processor = ImageProcessor()
        self.pdf_handler = PDFHandler()
        self.image_handler = ImageHandler()
        self.label_handler = LabelHandler()

    @staticmethod
    def _create_images_labels_dict(images_dir: str,
                                   labels_dir: str) -> Dict[str, str]:
        # List of all images and labels in directory
        images_listdir = os.listdir(images_dir)
        labels_listdir = os.listdir(labels_dir)

        # Create a dictionary to store the images and labels names
        images_labels = {}
        for image_name in images_listdir:
            label_name = os.path.splitext(image_name)[0] + '.txt'

            if label_name in labels_listdir:
                images_labels[image_name] = label_name
            else:
                images_labels[image_name] = None

        return images_labels

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

        # Images and labels
        images_labels = DataCreator._create_images_labels_dict(images_dir=images_dir,
                                                               labels_dir=labels_dir)

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
                                                                         images_labels=images_labels,
                                                                         classes_dict=classes_dict,
                                                                         target_classes=["question"])
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

    def predict_questions(self,
                          raw_dir: str,
                          train_dir: str,
                          yolo_model_path: str,
                          scan_type: str,
                          num_images: int) -> None:

        # Load model and labels
        yolo_predictor = YoloPredictor(model_path=yolo_model_path)

        # Pdf listdir
        pdf_listdir = [pdf for pdf in os.listdir(
            raw_dir) if pdf.endswith('pdf')]

        # Counter for saved images
        num_saved = 0

        while num_images != num_saved:
            # Extract random image, points
            rand_image, rand_image_name, rand_page_idx = self.pdf_handler.get_random_image(pdf_listdir=pdf_listdir,
                                                                                           pdf_dir=raw_dir)
            rand_image = self.image_processor.process(image=rand_image,
                                                      scan_type=scan_type)

            points_dict = yolo_predictor.get_points(image=rand_image)
            rand_points_idx, rand_points = self.label_handler._get_random_points(classes_dict=yolo_predictor.classes_dict,
                                                                                 points_dict=points_dict,
                                                                                 target_classes=["question"])

            # Crop question image and add borders
            rand_image = self.image_handler.crop_image(image=rand_image,
                                                       points=rand_points)

            # Save image
            num_saved = self._save_image(rand_page_idx,
                                         rand_points_idx,
                                         train_dir=train_dir,
                                         image=rand_image,
                                         image_name=rand_image_name,
                                         num_saved=num_saved,
                                         num_images=num_images)

    # def create_ocr_train_data_raw(self,
    #                               raw_dir: str,
    #                               train_dir: str,
    #                               target_classes: list[str],
    #                               num_images: int) -> None:
    #     # Load detection model and preprocessor
    #     text_det_processor = load_text_det_processor()
    #     text_det_model = load_text_det_model()

    #     # Paths
    #     images_dir = os.path.join(raw_dir, "images")
    #     labels_dir = os.path.join(raw_dir, "labels")
    #     classes_path = os.path.join(raw_dir, "classes.txt")

    #     # Classes
    #     classes = DataCreator.__read_classes_file(classes_path)

    #     # Images and labels
    #     images_labels = DataCreator.__create_images_labels_dict(images_dir=images_dir,
    #                                                             labels_dir=labels_dir)

    #     # Images listdir
    #     images_listdir = os.listdir(images_dir)

    #     # Counter for saved images
    #     num_saved = 0

    #     while num_images != num_saved:
    #         rand_image_name = random.choice(images_listdir)
    #         rand_image_path = os.path.join(images_dir, rand_image_name)
    #         rand_label_name = images_labels[rand_image_name]
    #         rand_label_path = os.path.join(labels_dir, rand_label_name)

    #         # Raise exception of label name is None
    #         if rand_label_name is None:
    #             raise ValueError("Label must not be None.")

    #         # Extract random points and crop corresponding image
    #         all_points = DataCreator.__get_question_points(label_path=rand_label_path,
    #                                                        classes=classes,
    #                                                        target_classes=target_classes)

    #         if not all_points:
    #             continue

    #         rand_points_index, rand_points = DataCreator.__get_random_points(
    #             all_points=all_points)

    #         # Crop question image and add borders
    #         rand_image = Image.open(rand_image_path)
    #         rand_image = DataCreator.__crop_image(image=rand_image,
    #                                               points=rand_points,
    #                                               offset=0)
    #         # Detect text
    #         rand_subimages = DataCreator.__detect_text(image=rand_image,
    #                                                    det_processor=text_det_processor,
    #                                                    det_model=text_det_model)

    #         for rand_subimage_index, rand_subimage in enumerate(rand_subimages):
    #             # Save image
    #             save_image_name = os.path.splitext(rand_image_name)[0]
    #             save_image_name = f"{save_image_name}_{
    #                 rand_points_index}_raw_{rand_subimage_index}.jpg"
    #             save_image_path = os.path.join(train_dir, save_image_name)

    #             if not os.path.exists(save_image_path):
    #                 rand_subimage.save(save_image_path, "JPEG")

    #         # Count images
    #         if len(rand_subimages) > 0:
    #             num_saved += 1
    #             print(f"It was saved {
    #                 num_saved}/{num_images} images.")

    # def create_ocr_train_data_pred(self,
    #                                raw_dir: str,
    #                                train_dir: str,
    #                                yolo_page_model_path: str,
    #                                yolo_question_model_path: str,
    #                                num_images: int) -> None:
    #     # Load model and labels
    #     text_det_processor = load_text_det_processor()
    #     text_det_model = load_text_det_model()
    #     page_seg_model = YOLO(yolo_page_model_path)
    #     question_seg_model = YOLO(yolo_question_model_path)
    #     page_labels = page_seg_model.names
    #     question_labels = question_seg_model.names

    #     # Pdf listdir
    #     pdf_listdir = [pdf for pdf in os.listdir(
    #         raw_dir) if pdf.endswith('pdf')]

    #     # Counter for saved images
    #     num_saved = 0

    #     while num_images != num_saved:
    #         # Get random image
    #         rand_page_image, rand_image_path = self.__get_random_image_from_pdf(pdf_listdir=pdf_listdir,
    #                                                                             raw_dir=raw_dir,
    #                                                                             train_dir=train_dir)

    #         # Create list to store all points with question
    #         all_page_points = []
    #         pred_question_points = []

    #         # Predict image and get points
    #         page_result = page_seg_model.predict(
    #             rand_page_image, verbose=False)[0]

    #         for page_box, page_points in zip(page_result.boxes, page_result.masks.xyn):

    #             # Get points and label
    #             page_points = page_points.reshape(-1).tolist()
    #             page_label = page_labels[int(page_box.cls.item())]

    #             if page_label != "question":
    #                 all_page_points.append(page_points)
    #             else:
    #                 pred_question_points.append(page_points)

    #         if not all_page_points:
    #             continue

    #         # Get random points
    #         rand_page_index, rand_page_points = DataCreator.__get_random_points(
    #             all_points=all_page_points)
    #         _, pred_question_points = DataCreator.__get_random_points(
    #             all_points=pred_question_points)

    #         # Crop question image and add borders
    #         rand_page_subimage = DataCreator.__crop_image(image=rand_page_image,
    #                                                       points=rand_page_points,
    #                                                       offset=0)

    #         question_image_to_predict = DataCreator.__crop_image(image=rand_page_image,
    #                                                              points=pred_question_points)

    #         # Predict questions
    #         question_result = question_seg_model.predict(
    #             question_image_to_predict, verbose=False)[0]

    #         all_question_points = []

    #         for question_box, question_points in zip(question_result.boxes, question_result.masks.xyn):

    #             # Get points and label
    #             question_points = question_points.reshape(-1).tolist()
    #             question_label = question_labels[int(question_box.cls.item())]

    #             if question_label != "image":
    #                 all_question_points.append(question_points)

    #         if not all_question_points:
    #             continue

    #         # Get random points
    #         rand_question_index, rand_question_points = DataCreator.__get_random_points(
    #             all_points=all_question_points)
    #         rand_question_image = DataCreator.__crop_image(image=question_image_to_predict,
    #                                                        points=rand_question_points,
    #                                                        offset=0)

    #         # Detect text
    #         rand_page_subimages = DataCreator.__detect_text(image=rand_page_subimage,
    #                                                         det_processor=text_det_processor,
    #                                                         det_model=text_det_model)
    #         rand_question_subimages = DataCreator.__detect_text(image=rand_question_image,
    #                                                             det_processor=text_det_processor,
    #                                                             det_model=text_det_model)

    #         # Save subimages
    #         base_save_image_name = os.path.splitext(
    #             os.path.basename(rand_image_path))[0]
    #         for rand_subimage_index, rand_subimage in enumerate(rand_page_subimages):
    #             save_image_name = f"{base_save_image_name}_pred_page_{
    #                 rand_subimage_index}_{rand_page_index}.jpg"
    #             save_image_path = os.path.join(train_dir, save_image_name)

    #             if not os.path.exists(save_image_path):
    #                 rand_subimage.save(save_image_path, "JPEG")

    #         for rand_subimage_index, rand_subimage in enumerate(rand_question_subimages):
    #             save_image_name = f"{base_save_image_name}_pred_page_{
    #                 rand_subimage_index}_{rand_question_index}.jpg"
    #             save_image_path = os.path.join(train_dir, save_image_name)

    #             if not os.path.exists(save_image_path):
    #                 rand_subimage.save(save_image_path, "JPEG")

    #         # Count images
    #         if len(rand_page_subimages) > 0:
    #             num_saved += 1
    #             print(f"It was saved {
    #                 num_saved}/{num_images} images.")
