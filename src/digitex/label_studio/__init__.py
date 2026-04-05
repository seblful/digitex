"""Label Studio integration."""

from digitex.label_studio.client import LabelStudioClient
from digitex.label_studio.predictor import TaskPredictor

__all__ = ["LabelStudioClient", "TaskPredictor"]
