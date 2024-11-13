from abc import ABC, abstractmethod


class Predictor(ABC):
    @property
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class YoloPredictor(Predictor):
    def __init__(self, model_path: str):
        self.model_path = model_path

    @property
    def model(self):
        pass

    def predict(self):
        pass
