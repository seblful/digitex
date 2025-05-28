from PIL import Image

import numpy as np
import torch

from openocr import OpenDetector
from openocr.tools.engine.config import Config

from .abstract_predictor import Predictor
from .prediction_result import DetectionPredictionResult


class DB_RepVitDetectionPredictor(Predictor):
    def __init__(self, config_path: str, model_path: str, device: torch.device) -> None:
        self.config_path = config_path
        self.model_path = model_path
        self.device = device

        self.__config = None
        self.__model = None

    @property
    def config(self) -> dict[str]:
        if self.__config is None:
            config = Config(self.config_path).cfg
            config["Global"]["pretrained_model"] = self.model_path
            config["Global"]["device"] = "gpu" if self.device == "cuda" else "cpu"

            self.__config = config

        return self.__config

    @property
    def model(self) -> OpenDetector:
        if self.__model is None:
            self.__model = OpenDetector(self.config)

        return self.__model

    def preprocess_image(self, image: Image) -> np.ndarray:
        return np.array(image)

    def create_result(self, preds: list[dict]) -> DetectionPredictionResult:
        ids = []
        points = []

        bboxes = preds[0]["boxes"]

        # Process bboxes, ids and append to lists
        for bbox in bboxes:
            bbox = bbox.astype(np.int32).flatten().tolist()
            ids.append(0)
            points.append(bbox)

        if len(ids) == 0:
            return None

        # Create result
        result = DetectionPredictionResult(ids=ids, points=points, id2label={0: "text"})

        return result

    def predict(self, image) -> DetectionPredictionResult:
        img = self.preprocess_image(image)
        preds = self.model(img_numpy=img)
        result = self.create_result(preds)
        return result
