from PIL import Image

import numpy as np
import torch

from doctr.models import detection_predictor, fast_base

from .abstract_predictor import Predictor
from .prediction_result import DetectionPredictionResult


class FAST_DetectionPredictor(Predictor):
    def __init__(self,
                 model_path: str) -> None:
        self.model_path = model_path
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.__model = None

    @property
    def model(self) -> detection_predictor:
        if self.__model is None:
            torch_model = fast_base(pretrained=False,
                                    pretrained_backbone=False)
            params = torch.load(self.model_path, map_location=self.device)
            torch_model.load_state_dict(params)

            model = detection_predictor(torch_model,
                                        pretrained=True,
                                        assume_straight_pages=False)
            self.__model = model

        return self.__model

    def preprocess_image(self,
                         image: Image) -> np.ndarray:
        return np.array(image)

    def create_result(self,
                      preds: list[dict],
                      img_width: int,
                      img_height: int) -> DetectionPredictionResult:

        ids = []
        points = []

        # Process bboxes, ids and append to lists
        bboxes = preds[0]["words"]
        for bbox in bboxes:
            bbox = bbox[:4] * np.array([img_width, img_height])
            bbox = bbox.astype(np.int32)
            bbox = bbox.flatten().tolist()

            ids.append(0)
            points.append(bbox)

        # Create result
        result = DetectionPredictionResult(ids=ids,
                                           points=points,
                                           id2label={0: "text"})

        return result

    def predict(self,
                image: Image) -> DetectionPredictionResult:
        img = self.preprocess_image(image)
        img_height, img_width, _ = img.shape
        preds = self.model([img])
        result = self.create_result(preds, img_width, img_height)

        return result
