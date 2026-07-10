"""ML predictors."""

from .prediction_result import SegmentationPredictionResult
from .segmentation import YOLO_SegmentationPredictor

__all__ = ["SegmentationPredictionResult", "YOLO_SegmentationPredictor"]
