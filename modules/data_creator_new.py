from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import os
import random

from PIL import Image

import cv2
import numpy as np

import pypdfium2 as pdfium

from ultralytics import YOLO
from transformers import PreTrainedModel, SegformerImageProcessor
# from surya.model.detection.model import load_model as load_text_det_model, load_processor as load_text_det_processor
# from surya.detection import batch_text_detection

from modules.processors import ImageProcessor


@dataclass
class ImageData:
    image: Image.Image
    path: str


class ImageUtils:
    @staticmethod
    def crop_image(image: Image.Image, points: List[float], offset: float = 0.025) -> Image.Image:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]

        pts = np.array([(int(x * width), int(y * height)) for x, y in points])
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        img = img[y:y+h, x:x+w].copy()

        pts = pts - pts.min(axis=0)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        result = cv2.bitwise_and(img, img, mask=mask)
        bg = np.ones_like(img, np.uint8)*255
        cv2.bitwise_not(bg, bg, mask=mask)
        result = bg + result

        border = int(height*offset)
        result = cv2.copyMakeBorder(result, border, border, border, border,
                                    cv2.BORDER_CONSTANT, value=[255, 255, 255])

        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


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

    @ staticmethod
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
