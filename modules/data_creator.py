from typing import List, Tuple, Dict

import os
import random
from PIL import Image

from ultralytics import YOLO

from transformers import PreTrainedModel, SegformerImageProcessor
# from surya.model.detection.model import load_model as load_text_det_model, load_processor as load_text_det_processor
# from surya.detection import batch_text_detection

from modules.processors import ImageProcessor, PDFHandler
from modules.utils import ImageUtils


class LabelHandler:
    def _read_classes(self, classes_path: str) -> Dict[int, str]:
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
            return {i: cl for i, cl in enumerate(classes)}

    def _read_points(self, label_path: str) -> Dict[int, list[list[float]]]:
        points_dict = dict()
        with open(label_path, "r") as f:
            for line in f:
                # Get points
                data = line.strip().split()
                class_idx = int(data[0])
                points = [float(point) for point in data[1:]]

                # Append points to the list in dict
                if class_idx not in points_dict:
                    points_dict[class_idx] = []
                points_dict[class_idx].append(points)

        return points_dict

    def _get_random_points(self,
                           classes_dict: Dict[int, str],
                           points_dict: Dict[int, list],
                           target_classes: List[str]) -> List[List[float]]:
        # Create subset of dict with target classes
        points_dict = {k: points_dict[k]
                       for k in points_dict if classes_dict[k] in target_classes}

        # Get random label
        rand_class_idx = random.choice(points_dict.keys())
        rand_label_name = classes_dict[rand_class_idx]

        # Get random points
        rand_points_idx = random.randint(
            0, len(points_dict[rand_class_idx]))
        rand_points = points_dict[rand_class_idx][rand_points_idx]

        return rand_label_name, rand_points_idx, rand_points

    def get_points(self,
                   classes_path: str,
                   label_path: str,
                   target_classes: List[str]) -> List[float]:
        classes_dict = self._read_classes(classes_path)
        points_dict = self._read_points(label_path)
        rand_label_name, rand_points_idx, rand_points = self._get_random_points(classes_dict=classes_dict,
                                                                                points_dict=points_dict,
                                                                                target_classes=target_classes)

        return rand_label_name, rand_points_idx, rand_points


class DataCreator:
    def __init__(self) -> None:
        self.image_processor = ImageProcessor()
        self.pdf_handler = PDFHandler()
        self.image_utils = ImageUtils()
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

    def _save_image(self,
                    *args,
                    train_dir: str,
                    image: Image.Image,
                    image_name: str,
                    num_saved: int,
                    num_images: int) -> int:

        # Create image path
        str_ids = "_".join([str(i) for i in args])
        image_path = os.path.join(
            train_dir, image_name) + "_" + str_ids + ".jpg"

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
                                                                                           raw_dir=raw_dir)

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

    def create_question_train_data_raw(self,
                                       page_raw_dir: str,
                                       train_dir: str,
                                       num_images: int) -> None:
        # Paths
        images_dir = os.path.join(page_raw_dir, "images")
        labels_dir = os.path.join(page_raw_dir, "labels")
        classes_path = os.path.join(page_raw_dir, "classes.txt")

        # Classes
        classes = DataCreator.__read_classes_file(classes_path)

        # Images and labels
        images_labels = DataCreator.__create_images_labels_dict(images_dir=images_dir,
                                                                labels_dir=labels_dir)

        # Images listdir
        images_listdir = os.listdir(images_dir)

        # Counter for saved images
        num_saved = 0

        while num_images != num_saved:
            rand_image_name = random.choice(images_listdir)
            rand_image_path = os.path.join(images_dir, rand_image_name)
            rand_label_name = images_labels[rand_image_name]
            rand_label_path = os.path.join(labels_dir, rand_label_name)

            # Raise exception of label name is None
            if rand_label_name is None:
                raise ValueError("Label must not be None.")

            # Extract random points and crop corresponding image
            all_points = DataCreator.__get_question_points(label_path=rand_label_path,
                                                           classes=classes)
            rand_points_index, rand_points = DataCreator.__get_random_points(
                all_points=all_points)

            # Crop question image and add borders
            rand_image = Image.open(rand_image_path)
            rand_image = DataCreator.__crop_image(image=rand_image,
                                                  points=rand_points)

            # Save image
            save_image_name = os.path.splitext(rand_image_name)[0]
            save_image_name = f"{save_image_name}_{rand_points_index}.jpg"
            save_image_path = os.path.join(train_dir, save_image_name)

            if not os.path.exists(save_image_path):
                rand_image.save(save_image_path, "JPEG")
                num_saved += 1
                print(f"It was saved {num_saved}/{num_images} images.")

    def create_question_train_data_pred(self,
                                        raw_dir: str,
                                        train_dir: str,
                                        yolo_model_path: str,
                                        num_images: int) -> None:

        # Load model and labels
        model = YOLO(yolo_model_path)
        labels = model.names

        # Pdf listdir
        pdf_listdir = [pdf for pdf in os.listdir(
            raw_dir) if pdf.endswith('pdf')]

        # Counter for saved images
        num_saved = 0

        while num_images != num_saved:
            # Get random image
            rand_image, rand_image_path = self.__get_random_image_from_pdf(pdf_listdir=pdf_listdir,
                                                                           raw_dir=raw_dir,
                                                                           train_dir=train_dir)

            # Create list to store all points with question
            all_points = []

            # Predict image and get points
            result = model.predict(rand_image, verbose=False)[0]

            for box, points in zip(result.boxes, result.masks.xyn):

                # Get points and label
                points = points.reshape(-1).tolist()
                label = labels[int(box.cls.item())]

                if label == "question":
                    all_points.append(points)

            # Get random points
            rand_points_index, rand_points = DataCreator.__get_random_points(
                all_points=all_points)

            # Crop question image and add borders
            rand_image = DataCreator.__crop_image(image=rand_image,
                                                  points=rand_points)

            # Save image
            save_image_name = os.path.splitext(
                os.path.basename(rand_image_path))[0]
            save_image_name = f"{save_image_name}_{rand_points_index}.jpg"
            save_image_path = os.path.join(train_dir, save_image_name)

            if not os.path.exists(save_image_path):
                rand_image.save(save_image_path, "JPEG")
                num_saved += 1
                print(f"It was saved {num_saved}/{num_images} images.")

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
