import numpy as np
from pathlib import Path

import structlog
import torch
from PIL import Image
from shapely.geometry import Polygon
from ultralytics import YOLO  # type: ignore[import-untyped]
from ultralytics.engine.results import Results

from digitex.utils import get_device

from .abstract_predictor import Predictor
from .prediction_result import SegmentationPredictionResult

logger = structlog.get_logger()


class YOLO_SegmentationPredictor(Predictor):
    """YOLO-based segmentation predictor for document analysis."""

    def __init__(
        self,
        model_path: str | Path,
        simplify: bool = False,
        epsilon: float = 2.0,
    ) -> None:
        """Initialize the YOLO segmentation predictor.

        Args:
            model_path: Path to the trained YOLO model file.
            simplify: Whether to apply Douglas-Peucker polygon simplification.
            epsilon: Distance tolerance for Douglas-Peucker algorithm (pixels).

        Raises:
            FileNotFoundError: If model_path doesn't exist.
        """
        self.model_path = model_path
        self.simplify = simplify
        self.epsilon = epsilon
        self.device = get_device()
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO:
        """Get or load the YOLO model.

        Returns:
            Loaded YOLO model.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        if self._model is None:
            try:
                self._model = YOLO(str(self.model_path), verbose=False)
                logger.info(f"Model loaded successfully from {self.model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

        return self._model

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess a PIL Image for YOLO prediction.

        Args:
            image: PIL Image to preprocess.

        Returns:
            NumPy array representation of the image.
        """
        img = np.array(image)
        return img

    def create_result(
        self,
        preds: list[Results],
        img_width: int,
        img_height: int,
    ) -> SegmentationPredictionResult:
        """Create a segmentation prediction result from YOLO predictions.

        Args:
            preds: Raw predictions from YOLO model.
            img_width: Original image width in pixels.
            img_height: Original image height in pixels.

        Returns:
            SegmentationPredictionResult containing IDs and polygons.

        Raises:
            ValueError: If predictions are invalid.
        """
        if not preds:
            raise ValueError("Empty predictions received")

        ids = []
        polygons = []

        pred = preds[0]
        if not hasattr(pred, "boxes") or not hasattr(pred, "masks"):
            raise ValueError("Invalid prediction format")

        boxes = pred.boxes
        masks = pred.masks

        if boxes is None or masks is None:
            logger.warning("No boxes or masks found in predictions")
            return SegmentationPredictionResult(
                ids=[], polygons=[], id2label=self.model.names
            )

        mask_data = masks.xyn

        for box, polygon in zip(boxes, mask_data):  # ty: ignore[invalid-argument-type]
            try:
                polygon = polygon * np.array([img_width, img_height])
                polygon = polygon.astype(np.int32)
                polygon = polygon.tolist()
                polygon = [tuple(points) for points in polygon]

                if self.simplify:
                    poly = Polygon(polygon)
                    simplified = poly.simplify(self.epsilon, preserve_topology=True)
                    polygon = [(int(x), int(y)) for x, y in simplified.exterior.coords]

                idx = int(box.cls.item())

                ids.append(idx)
                polygons.append(polygon)
            except Exception as e:
                logger.warning(f"Failed to process prediction: {e}")
                continue

        result = SegmentationPredictionResult(
            ids=ids, polygons=polygons, id2label=self.model.names
        )

        return result

    def predict(self, image: Image.Image) -> SegmentationPredictionResult:
        """Make a segmentation prediction on an image.

        Args:
            image: PIL Image to predict on.

        Returns:
            SegmentationPredictionResult containing IDs and polygons.
        """
        img = self.preprocess_image(image)
        img_height, img_width, _ = img.shape

        preds = self.model.predict(img, verbose=False)
        result = self.create_result(preds, img_width, img_height)

        return result
