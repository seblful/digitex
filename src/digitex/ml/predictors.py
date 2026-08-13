"""YOLO segmentation — page image in, detections out.

``detections_from`` is a pure function over YOLO's ``Results`` and a class map,
so it can be exercised without loading a model. ``YOLO_SegmentationPredictor``
adds the lazily loaded model and nothing else.
"""

from pathlib import Path

import numpy as np
import structlog
import torch
from PIL import Image
from shapely.geometry import Polygon
from ultralytics import YOLO  # type: ignore[import-untyped]
from ultralytics.engine.results import Results

from digitex.core.domain import Detection, PixelPolygon

logger = structlog.get_logger()

# Douglas-Peucker tolerance, in source-image pixels.
SIMPLIFY_EPSILON = 3.0


def _simplify(polygon: PixelPolygon, epsilon: float = SIMPLIFY_EPSILON) -> PixelPolygon:
    """Drop points a Douglas-Peucker pass finds redundant."""
    simplified = Polygon(polygon).simplify(epsilon, preserve_topology=True)
    return PixelPolygon([(int(x), int(y)) for x, y in simplified.exterior.coords])


def detections_from(
    preds: list[Results],
    img_width: int,
    img_height: int,
    id2label: dict[int, str],
    *,
    simplify: bool = False,
) -> list[Detection]:
    """Turn one YOLO prediction into detections in source-image pixels.

    YOLO reports masks normalized to 0-1; each is scaled back up by the image
    size it was predicted on. A detection whose mask cannot be processed is
    logged and skipped rather than failing the page.

    Raises:
        ValueError: If *preds* is empty or the first prediction has no
            ``boxes`` / ``masks`` attributes.
    """
    if not preds:
        raise ValueError("Empty predictions received")

    pred = preds[0]
    if not hasattr(pred, "boxes") or not hasattr(pred, "masks"):
        raise ValueError("Invalid prediction format")

    boxes = pred.boxes
    masks = pred.masks
    if boxes is None or masks is None:
        logger.warning("No boxes or masks found in predictions")
        return []

    mask_data = masks.xyn
    if len(boxes) != len(mask_data):
        logger.warning(
            "Box and mask counts differ, pairing only what lines up",
            boxes=len(boxes),
            masks=len(mask_data),
        )

    detections: list[Detection] = []
    dropped = 0
    # Boxes indexes but does not iterate, so pair the two by position.
    for i in range(min(len(boxes), len(mask_data))):
        box, raw_polygon = boxes[i], mask_data[i]
        try:
            scaled = raw_polygon * np.array([img_width, img_height])
            polygon = PixelPolygon([tuple(p) for p in scaled.astype(np.int32).tolist()])
            if simplify:
                polygon = _simplify(polygon)
            class_id = int(box.cls.item())
            detections.append(
                Detection(label=id2label.get(class_id, "unknown"), polygon=polygon)
            )
        except Exception:
            # A dropped marker silently re-files the rest of a book under the
            # wrong option, so say how many were lost rather than just that one
            # was — and keep the traceback.
            dropped += 1
            logger.warning("Failed to process prediction", index=i, exc_info=True)

    if dropped:
        logger.warning(
            "Dropped detections on this page", dropped=dropped, kept=len(detections)
        )

    return detections


class YOLO_SegmentationPredictor:
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
        """
        self.model_path = model_path
        self.simplify = simplify
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO:
        """Get or load the YOLO model.

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
                logger.info("Model loaded successfully", model_path=self.model_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model from {self.model_path}: {e}"
                ) from e

        return self._model

    def predict(
        self,
        image: Image.Image,
        conf: float = 0.25,
        imgsz: int | list[int] = 640,
        end2end: bool = False,
        agnostic_nms: bool = True,
        verbose: bool = False,
    ) -> list[Detection]:
        """Detect labelled regions on *image*, in source-image pixels.

        Args:
            image: PIL Image to predict on.
            conf: Confidence threshold for predictions (0.0-1.0).
            imgsz: Image size for inference (int or list).
            end2end: Whether to use end-to-end mode (removes NMS).
            agnostic_nms: Whether to use agnostic NMS.
            verbose: Whether to enable verbose output.
        """
        img_width, img_height = image.size

        preds = self.model.predict(
            image,
            conf=conf,
            imgsz=imgsz,
            end2end=end2end,
            agnostic_nms=agnostic_nms,
            verbose=verbose,
        )
        return detections_from(
            preds,
            img_width,
            img_height,
            dict(self.model.names),
            simplify=self.simplify,
        )
