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
