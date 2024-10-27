from typing import List, Tuple, Dict

import os
import random
from PIL import Image

import cv2
import numpy as np
import pypdfium2 as pdfium

from ultralytics import YOLO

from transformers import PreTrainedModel, SegformerImageProcessor
from surya.model.detection.model import load_model as load_text_det_model, load_processor as load_text_det_processor
from surya.detection import batch_text_detection

from modules.processors import ImageProcessor


class DataCreator:
    def __init__(self) -> None:

        # Image processor
        self.image_processor = ImageProcessor()

    def create_pdf_from_images(self,
                               image_dir: str,
                               raw_dir: str,
                               process: bool = False) -> None:
        # Create new pdf object
        pdf = pdfium.PdfDocument.new()

        # Iterate through images
        for image_name in sorted(os.listdir(image_dir),
                                 key=lambda x: int(x.split("_")[-1].split(".")[0])):
            # Load image
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path)

            # Process image
            if process is True:
                image = self.image_processor.process(
                    image=image, scan_type="color")

            bitmap = pdfium.PdfBitmap.from_pil(image)
            pdf_image = pdfium.PdfImage.new(pdf)

            pdf_image.set_bitmap(bitmap)

            image.close()
            bitmap.close()

            # Create, scale and set_matrix
            width, height = pdf_image.get_size()
            matrix = pdfium.PdfMatrix().scale(width, height)
            pdf_image.set_matrix(matrix)

            # Create page and insert image to it
            page = pdf.new_page(width, height)
            page.insert_obj(pdf_image)
            page.gen_content()

        # Save pdf
        images_dir_name = os.path.basename(image_dir)
        raw_dir_name = os.path.basename(raw_dir)
        pdf_name = images_dir_name + " " + raw_dir_name + ".pdf"
        pdf_path = os.path.join(raw_dir, pdf_name)
        pdf.save(pdf_path, version=17)

    def __get_page_image(self,
                         page: pdfium.PdfPage,
                         scale: int = 3) -> Image.Image:
        # Get image from pdf
        bitmap = page.render(scale=scale,
                             rotation=0)
        image = bitmap.to_pil()

        # Check image mode and convert if not RGB
        image_mode = image.mode

        if image_mode != 'RGB':
            image = image.convert('RGB')

        return image

    def __get_random_image_from_pdf(self,
                                    pdf_listdir: list[str],
                                    raw_dir: str,
                                    train_dir: str) -> Tuple[Image.Image | str]:
        # Take random pdf
        rand_pdf_name = random.choice(pdf_listdir)
        rand_pdf_path = os.path.join(raw_dir, rand_pdf_name)
        rand_pdf_obj = pdfium.PdfDocument(rand_pdf_path)

        # Take random pdf page and image
        rand_page_ind = random.randint(0, len(rand_pdf_obj) - 1)
        rand_page = rand_pdf_obj[rand_page_ind]

        # Get random image and preprocess it
        rand_image = self.__get_page_image(page=rand_page)

        save_pdf_name = os.path.splitext(rand_pdf_name)[0]
        rand_image_name = f"{save_pdf_name}_{rand_page_ind}.jpg"
        rand_image_path = os.path.join(train_dir, rand_image_name)

        # Close pdf file-object
        rand_pdf_obj.close()

        return rand_image, rand_image_path

    def create_yolo_train_data(self,
                               raw_dir: str,
                               train_dir: str,
                               scan_type: str,
                               num_images: int) -> None:

        # Pdf listdir
        pdf_listdir = [pdf for pdf in os.listdir(
            raw_dir) if pdf.endswith('pdf')]

        # Counter for saved images
        num_saved_images = 0

        while num_images != num_saved_images:

            rand_image, rand_image_path = self.__get_random_image_from_pdf(pdf_listdir=pdf_listdir,
                                                                           raw_dir=raw_dir,
                                                                           train_dir=train_dir)
            # Process image
            rand_image = self.image_processor.process(image=rand_image,
                                                      scan_type=scan_type)

            if not os.path.exists(rand_image_path):
                rand_image.save(rand_image_path, "JPEG")
                num_saved_images += 1
                print(f"It was saved {num_saved_images}/{num_images} images.")

            rand_image.close()

    @staticmethod
    def __read_classes_file(classes_path) -> Dict[int, str]:
        with open(classes_path, 'r') as classes_file:
            # Set the names of the classes
            classes = [i.split('\n')[0] for i in classes_file.readlines()]
            classes = {i: cl for i, cl in enumerate(classes)}

        return classes

    @staticmethod
    def __create_images_labels_dict(images_dir: str,
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
    def __get_question_points(label_path: str,
                              classes: Dict[int, str],
                              target_classes: list[str] = ["question"]) -> List[Tuple[float, float]]:
        # Create list to store all points with question
        all_points = []

        # Open label
        with open(label_path, "r") as file:
            for line in file.readlines():
                # Retrieve data
                data = line.strip("\n").split()
                label = int(data[0])
                points = [float(point) for point in data[1:]]

                # If label is in targat_classes, store in the list
                if classes[label] in target_classes:
                    all_points.append(points)

        return all_points

    @staticmethod
    def __get_random_points(all_points: List[List[float]]) -> Tuple[int | List[Tuple[float]]]:
        # Find random point
        rand_points_index = random.randint(0, len(all_points) - 1)
        rand_points = all_points[rand_points_index]
        # Convert points to tuples
        rand_points = list(zip(rand_points[::2], rand_points[1::2]))

        return rand_points_index, rand_points

    @staticmethod
    def __crop_image(image: Image.Image,
                     points: List[float],
                     offset: float = 0.025) -> Image.Image:
        # Open image
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]

        # Convert points
        pts = np.array([(int(x * width), int(y * height)) for x, y in points])

        # Find rect of polygon
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        img = img[y:y+h, x:x+w].copy()

        # Create mask
        pts = pts - pts.min(axis=0)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        # Bitwise and with mask
        result = cv2.bitwise_and(img, img, mask=mask)

        # Add white background
        bg = np.ones_like(img, np.uint8)*255
        cv2.bitwise_not(bg, bg, mask=mask)
        result = bg + result

        # Add frame
        result = cv2.copyMakeBorder(result, int(height*offset), int(height*offset), int(width*offset), int(width*offset),
                                    cv2.BORDER_CONSTANT, value=[255, 255, 255])

        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    @staticmethod
    def __detect_text(image: Image.Image,
                      det_processor: SegformerImageProcessor,
                      det_model: PreTrainedModel) -> list[Image.Image]:
        det_result = batch_text_detection(images=[image],
                                          processor=det_processor,
                                          model=det_model)
        # Convert image to numpy
        img = np.array(image)

        # Iterate through detected bboxes and add cropped images to list
        det_images = []
        for polygon_box in det_result[0].bboxes:
            x_min, y_min, x_max, y_max = polygon_box.bbox

            det_img = img[y_min:y_max, x_min:x_max]

            if not np.sum(det_img) == 0:
                det_image = Image.fromarray(det_img)
                det_images.append(det_image)

        return det_images

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
        num_saved_images = 0

        while num_images != num_saved_images:
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
                num_saved_images += 1
                print(f"It was saved {num_saved_images}/{num_images} images.")

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
        num_saved_images = 0

        while num_images != num_saved_images:
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
                num_saved_images += 1
                print(f"It was saved {num_saved_images}/{num_images} images.")

    def create_ocr_train_data_raw(self,
                                  raw_dir: str,
                                  train_dir: str,
                                  target_classes: list[str],
                                  num_images: int) -> None:
        # Load detection model and preprocessor
        text_det_processor = load_text_det_processor()
        text_det_model = load_text_det_model()

        # Paths
        images_dir = os.path.join(raw_dir, "images")
        labels_dir = os.path.join(raw_dir, "labels")
        classes_path = os.path.join(raw_dir, "classes.txt")

        # Classes
        classes = DataCreator.__read_classes_file(classes_path)

        # Images and labels
        images_labels = DataCreator.__create_images_labels_dict(images_dir=images_dir,
                                                                labels_dir=labels_dir)

        # Images listdir
        images_listdir = os.listdir(images_dir)

        # Counter for saved images
        num_saved_images = 0

        while num_images != num_saved_images:
            rand_image_name = random.choice(images_listdir)
            rand_image_path = os.path.join(images_dir, rand_image_name)
            rand_label_name = images_labels[rand_image_name]
            rand_label_path = os.path.join(labels_dir, rand_label_name)

            # Raise exception of label name is None
            if rand_label_name is None:
                raise ValueError("Label must not be None.")

            # Extract random points and crop corresponding image
            all_points = DataCreator.__get_question_points(label_path=rand_label_path,
                                                           classes=classes,
                                                           target_classes=target_classes)

            if not all_points:
                continue

            rand_points_index, rand_points = DataCreator.__get_random_points(
                all_points=all_points)

            # Crop question image and add borders
            rand_image = Image.open(rand_image_path)
            rand_image = DataCreator.__crop_image(image=rand_image,
                                                  points=rand_points,
                                                  offset=0)
            # Detect text
            rand_subimages = DataCreator.__detect_text(image=rand_image,
                                                       det_processor=text_det_processor,
                                                       det_model=text_det_model)

            for rand_subimage_index, rand_subimage in enumerate(rand_subimages):
                # Save image
                save_image_name = os.path.splitext(rand_image_name)[0]
                save_image_name = f"{save_image_name}_{
                    rand_points_index}_raw_{rand_subimage_index}.jpg"
                save_image_path = os.path.join(train_dir, save_image_name)

                if not os.path.exists(save_image_path):
                    rand_subimage.save(save_image_path, "JPEG")

            # Count images
            if len(rand_subimages) > 0:
                num_saved_images += 1
                print(f"It was saved {
                    num_saved_images}/{num_images} images.")

    def create_ocr_train_data_pred(self,
                                   raw_dir: str,
                                   train_dir: str,
                                   yolo_page_model_path: str,
                                   yolo_question_model_path: str,
                                   num_images: int) -> None:
        # Load model and labels
        text_det_processor = load_text_det_processor()
        text_det_model = load_text_det_model()
        page_seg_model = YOLO(yolo_page_model_path)
        question_seg_model = YOLO(yolo_question_model_path)
        page_labels = page_seg_model.names
        question_labels = question_seg_model.names

        # Pdf listdir
        pdf_listdir = [pdf for pdf in os.listdir(
            raw_dir) if pdf.endswith('pdf')]

        # Counter for saved images
        num_saved_images = 0

        while num_images != num_saved_images:
            # Get random image
            rand_page_image, rand_image_path = self.__get_random_image_from_pdf(pdf_listdir=pdf_listdir,
                                                                                raw_dir=raw_dir,
                                                                                train_dir=train_dir)

            # Create list to store all points with question
            all_page_points = []
            pred_question_points = []

            # Predict image and get points
            page_result = page_seg_model.predict(
                rand_page_image, verbose=False)[0]

            for page_box, page_points in zip(page_result.boxes, page_result.masks.xyn):

                # Get points and label
                page_points = page_points.reshape(-1).tolist()
                page_label = page_labels[int(page_box.cls.item())]

                if page_label != "question":
                    all_page_points.append(page_points)
                else:
                    pred_question_points.append(page_points)

            if not all_page_points:
                continue

            # Get random points
            rand_page_index, rand_page_points = DataCreator.__get_random_points(
                all_points=all_page_points)
            _, pred_question_points = DataCreator.__get_random_points(
                all_points=pred_question_points)

            # Crop question image and add borders
            rand_page_subimage = DataCreator.__crop_image(image=rand_page_image,
                                                          points=rand_page_points,
                                                          offset=0)

            question_image_to_predict = DataCreator.__crop_image(image=rand_page_image,
                                                                 points=pred_question_points)

            # Predict questions
            question_result = question_seg_model.predict(
                question_image_to_predict, verbose=False)[0]

            all_question_points = []

            for question_box, question_points in zip(question_result.boxes, question_result.masks.xyn):

                # Get points and label
                question_points = question_points.reshape(-1).tolist()
                question_label = question_labels[int(question_box.cls.item())]

                if question_label != "image":
                    all_question_points.append(question_points)

            if not all_question_points:
                continue

            # Get random points
            rand_question_index, rand_question_points = DataCreator.__get_random_points(
                all_points=all_question_points)
            rand_question_image = DataCreator.__crop_image(image=question_image_to_predict,
                                                           points=rand_question_points,
                                                           offset=0)

            # Detect text
            rand_page_subimages = DataCreator.__detect_text(image=rand_page_subimage,
                                                            det_processor=text_det_processor,
                                                            det_model=text_det_model)
            rand_question_subimages = DataCreator.__detect_text(image=rand_question_image,
                                                                det_processor=text_det_processor,
                                                                det_model=text_det_model)

            # Save subimages
            base_save_image_name = os.path.splitext(
                os.path.basename(rand_image_path))[0]
            for rand_subimage_index, rand_subimage in enumerate(rand_page_subimages):
                save_image_name = f"{base_save_image_name}_pred_page_{
                    rand_subimage_index}_{rand_page_index}.jpg"
                save_image_path = os.path.join(train_dir, save_image_name)

                if not os.path.exists(save_image_path):
                    rand_subimage.save(save_image_path, "JPEG")

            for rand_subimage_index, rand_subimage in enumerate(rand_question_subimages):
                save_image_name = f"{base_save_image_name}_pred_page_{
                    rand_subimage_index}_{rand_question_index}.jpg"
                save_image_path = os.path.join(train_dir, save_image_name)

                if not os.path.exists(save_image_path):
                    rand_subimage.save(save_image_path, "JPEG")

            # Count images
            if len(rand_page_subimages) > 0:
                num_saved_images += 1
                print(f"It was saved {
                    num_saved_images}/{num_images} images.")
