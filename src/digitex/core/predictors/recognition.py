from PIL import Image

import numpy as np
import torch

from openocr import OpenRecognizer
from openocr.tools.engine.config import Config

from .abstract_predictor import Predictor
from .prediction_result import RecognitionPredictionResult


class SVTR2_RecognitionPredictor(Predictor):
    def __init__(
        self, config_path: str, model_path: str, charset_path: str, device: torch.device
    ) -> None:
        self.config_path = config_path
        self.model_path = model_path
        self.charset_path = charset_path
        self.device = device

        self.__config = None
        self.__model = None

    @property
    def config(self) -> dict[str]:
        if self.__config is None:
            config = Config(self.config_path).cfg
            config["Global"]["pretrained_model"] = self.model_path
            config["Global"]["device"] = "gpu" if self.device == "cuda" else "cpu"
            config["Global"]["character_dict_path"] = self.charset_path

            self.__config = config

        return self.__config

    @property
    def model(self) -> OpenRecognizer:
        if self.__model is None:
            self.__model = OpenRecognizer(self.config)
        return self.__model

    def preprocess_image(self, image: Image) -> np.ndarray:
        return image

    def create_result(self, preds: list[dict]) -> RecognitionPredictionResult:
        text = preds[0]["text"]
        probability = preds[0]["score"]

        result = RecognitionPredictionResult(
            text=text, probability=probability, id2label={0: "text"}
        )

        return result

    def predict(self, image) -> RecognitionPredictionResult:
        img = self.preprocess_image(image)
        preds = self.model(img_numpy=img)
        result = self.create_result(preds)
        return result
