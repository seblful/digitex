from PIL import Image

import numpy as np
import torch

from ultralytics import YOLO

from .abstract_predictor import Predictor
from .prediction_result import SegmentationPredictionResult


class YOLO_SegmentationPredictor(Predictor):
    def __init__(self, model_path: str, device: torch.device) -> None:
        self.model_path = model_path
        self.device = device
        self.__model = None

    @property
    def model(self) -> YOLO:
        if self.__model is None:
            self.__model = YOLO(self.model_path, verbose=False)

        return self.__model

    def preprocess_image(self, image: Image) -> np.ndarray:
        img = np.array(image)
        return img

    def create_result(
        self, preds: list[dict], img_width: int, img_height: int
    ) -> SegmentationPredictionResult:
        ids = []
        polygons = []

        # Process bboxes, ids and append to lists
        for box, polygon in zip(preds[0].boxes, preds[0].masks.xyn):
            polygon = polygon * np.array([img_width, img_height])
            polygon = polygon.astype(np.int32)
            polygon = polygon.tolist()
            polygon = [tuple(points) for points in polygon]

            idx = int(box.cls.item())

            ids.append(idx)
            polygons.append(polygon)

        # Create result
        result = SegmentationPredictionResult(
            ids=ids, polygons=polygons, id2label=self.model.names
        )

        return result

    def predict(self, image: Image.Image) -> SegmentationPredictionResult:
        img = self.preprocess_image(image)
        img_height, img_width, _ = img.shape
        preds = self.model.predict(img, verbose=False)
        result = self.create_result(preds, img_width, img_height)

        return result
