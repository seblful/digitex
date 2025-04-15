from abc import ABC, abstractmethod
from typing import Any
from PIL import Image


class Predictor(ABC):
    @property
    @abstractmethod
    def model(self) -> Any:
        pass

    @abstractmethod
    def preprocess_image(self,
                         image: Image) -> Any:
        pass

    @abstractmethod
    def predict(self, image: Image) -> Any:
        pass

    @abstractmethod
    def create_result(self) -> Any:
        pass

    def __call__(self, image: Image) -> Any:
        return self.predict(image)
