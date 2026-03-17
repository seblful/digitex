from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class Predictor(ABC):
    """Abstract base class for prediction models."""

    @property
    @abstractmethod
    def model(self) -> Any:
        """Get the underlying model instance.

        Returns:
            The model object.
        """
        pass

    @abstractmethod
    def preprocess_image(self, image: Image.Image) -> Any:
        """Preprocess an image for prediction.

        Args:
            image: PIL Image to preprocess.

        Returns:
            Preprocessed image in the format expected by the model.
        """
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Any:
        """Make a prediction on an image.

        Args:
            image: PIL Image to predict on.

        Returns:
            Prediction result.
        """
        pass

    @abstractmethod
    def create_result(self, *args: Any, **kwargs: Any) -> Any:
        """Create a prediction result from model output.

        Args:
            *args: Variable positional arguments for result creation.
            **kwargs: Variable keyword arguments for result creation.

        Returns:
            Formatted prediction result.
        """
        pass

    def __call__(self, image: Image.Image) -> Any:
        """Make a prediction on an image.

        Args:
            image: PIL Image to predict on.

        Returns:
            Prediction result.
        """
        return self.predict(image)
