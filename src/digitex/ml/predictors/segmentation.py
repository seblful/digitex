from pathlib import Path

import numpy as np
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
    ) -> None:
        """Initialize the YOLO segmentation predictor.

        Args:
            model_path: Path to the trained YOLO model file.
            simplify: Whether to apply Douglas-Peucker polygon simplification.

        Raises:
            FileNotFoundError: If model_path doesn't exist.
        """
        self.model_path = model_path
        self.simplify = simplify
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
                model_path = Path(self.model_path)
                if not model_path.is_absolute():
                    model_path = Path.cwd() / model_path
                model_str = str(model_path.resolve())
                self._model = YOLO(model_str, verbose=False)
                logger.info(f"Model loaded successfully from {self.model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

        return self._model

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess a PIL Image for YOLO prediction.

        Args:
            image: PIL Image to preprocess.

        Returns:
            The same image (YOLO accepts PIL Images directly).
        """
        return image

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
                    polygon = self._simplify_polygon(polygon)

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

    def _simplify_polygon(
        self,
        polygon: list[tuple[int, int]],
        epsilon: float = 3.0,
    ) -> list[tuple[int, int]]:
        """Simplify a polygon using the Douglas-Peucker algorithm.

        Args:
            polygon: List of (x, y) coordinate tuples.
            epsilon: Distance tolerance for simplification (pixels).

        Returns:
            Simplified polygon as list of (x, y) coordinate tuples.
        """
        poly = Polygon(polygon)
        simplified = poly.simplify(epsilon, preserve_topology=True)
        return [(int(x), int(y)) for x, y in simplified.exterior.coords]

    def predict(
        self,
        image: Image.Image,
        conf: float = 0.25,
        imgsz: int | list[int] = 640,
        end2end: bool = True,
        verbose: bool = False,
    ) -> SegmentationPredictionResult:
        """Make a segmentation prediction on an image.

        Args:
            image: PIL Image to predict on.
            conf: Confidence threshold for predictions (0.0-1.0).
            imgsz: Image size for inference (int or list).
            end2end: Whether to use end-to-end mode (removes NMS).
            verbose: Whether to enable verbose output.

        Returns:
            SegmentationPredictionResult containing IDs and polygons.
        """
        img_width, img_height = image.size
        preprocessed = self.preprocess_image(image)

        preds = self.model.predict(
            preprocessed,
            verbose=verbose,
            conf=conf,
            imgsz=imgsz,
            end2end=end2end,
        )
        result = self.create_result(preds, img_width, img_height)

        return result
