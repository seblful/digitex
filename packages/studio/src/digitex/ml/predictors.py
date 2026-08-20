"""YOLO segmentation inference — a page image in, labelled polygons out.

Two pieces that come apart on purpose. ``detections_from`` is a pure function
over ultralytics' ``Results`` and a class map, so every rule about what counts
as a usable region is exercisable with no checkpoint and no GPU.
``YOLO_SegmentationPredictor`` wraps it in a lazily loaded model and the pinned
predict settings, and holds no other state.

What a region *means* is not here. Reading order, numbering and cropping belong
to :mod:`digitex.pipeline`; the polygons leave in source-image pixels so nothing
downstream has to know what size the model ran at.
"""

import os
import pathlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import structlog
import torch
from PIL import Image
from shapely.geometry import Polygon
from ultralytics import YOLO  # type: ignore[import-untyped]
from ultralytics.engine.results import Results

from digitex.domain.entities import Detection, PixelPolygon

logger = structlog.get_logger()

# Douglas-Peucker tolerance, in source-image pixels.
SIMPLIFY_EPSILON = 3.0

# One detection recipe for every caller — page extraction, Label Studio
# pre-annotation and the tuning tool all have to see the same regions, or the
# model that was tuned is not the model that runs.
PREDICT_CONF = 0.25
PREDICT_IMGSZ = 640
# The same ``max_det`` the train and val configs carry. Nothing has come near
# the ceiling: the most regions found on any of the 647 pooled pages is 15.
PREDICT_MAX_DET = 50

# YOLO26 heads are NMS-free by default and this one is not good enough at it.
# Measured over the whole pool, letting the one2one head speak for itself turns
# 28 overlapping region pairs into 283, across a third of the pages, including
# pairs at IoU 1.0 — duplicates an annotator would have to delete by hand. So
# inference runs the one2many branch behind ordinary NMS, which the val config
# is now pinned to as well: validation scores the path that actually runs.
PREDICT_END2END = False
# Class-agnostic, so NMS keeps one region per anchor instead of one per class
# and a marker cannot come back a second time wearing another label.
PREDICT_AGNOSTIC_NMS = True


# The modules a pickle can name the concrete path classes under. 3.13 moved
# them into ``pathlib._local`` and re-exported them, and a checkpoint names
# whichever module the Python that wrote it called home — so both spellings
# have to answer.
_PATH_CLASS_MODULES = ("pathlib", "pathlib._local")


@contextmanager
def foreign_paths_readable() -> Iterator[None]:
    """Make a checkpoint pickled on the other platform loadable on this one.

    A YOLO checkpoint carries the paths of the run that produced it, and pickle
    rebuilds each one by calling the class it was saved as. A ``PosixPath``
    cannot be instantiated on Windows, so a model trained on Linux fails to
    load here with "cannot instantiate 'PosixPath' on your system" — before a
    single image is looked at.

    Nothing in inference reads those paths, so pointing the foreign name at the
    local flavour for the duration of the load is enough. Only the direction
    that is actually broken is patched: ``Path`` picks the *local* class, so the
    name being redirected is never the one it resolves to.
    """
    foreign = "PosixPath" if os.name == "nt" else "WindowsPath"
    # Both halves of the pair are read before anything is patched, so the
    # restore below puts back the class that was there rather than a stand-in.
    patched = [
        (module, getattr(module, foreign))
        for name in _PATH_CLASS_MODULES
        if (module := sys.modules.get(name)) is not None and hasattr(module, foreign)
    ]

    for module, _ in patched:
        setattr(module, foreign, pathlib.Path)
    try:
        yield
    finally:
        for module, original in patched:
            setattr(module, foreign, original)


def _simplify(polygon: PixelPolygon) -> PixelPolygon:
    """Drop the points a Douglas-Peucker pass finds redundant."""
    ring = Polygon(polygon).simplify(SIMPLIFY_EPSILON, preserve_topology=True)
    return PixelPolygon([(int(x), int(y)) for x, y in ring.exterior.coords])


def _detection(
    box: Any,
    outline: np.ndarray,
    img_width: int,
    img_height: int,
    id2label: dict[int, str],
    *,
    simplify: bool,
) -> Detection | None:
    """One box and its mask as a detection, or ``None`` if it holds no ring.

    *box* is one row of ultralytics' ``Boxes``, which the vendor leaves
    unannotated — there is nothing narrower than ``Any`` to say about it here.
    *outline* is the matching mask, normalized to 0-1 and scaled back up by the
    size of the image it was predicted on.

    A mask with no contour above the threshold comes back from ultralytics as an
    empty (0, 2) array, which raises nothing. Two points short of a ring is not
    a region either: it uploads to Label Studio as a pointless polygon, and page
    extraction dies asking a reading order for the min() of nothing. Anything
    under three points therefore leaves as ``None``.
    """
    scaled = outline * np.array([img_width, img_height])
    polygon = PixelPolygon([tuple(p) for p in scaled.astype(np.int32).tolist()])
    if simplify:
        polygon = _simplify(polygon)
    if len(polygon) < 3:
        return None

    return Detection(
        label=id2label.get(int(box.cls.item()), "unknown"),
        polygon=polygon,
        score=float(box.conf.item()),
    )


def detections_from(
    preds: list[Results],
    img_width: int,
    img_height: int,
    id2label: dict[int, str],
    *,
    simplify: bool = False,
) -> list[Detection]:
    """Turn one YOLO prediction into detections in source-image pixels.

    A single unusable mask never fails the page: whatever cannot be turned into
    a polygon is counted and reported, and the rest of the page is returned.

    Args:
        preds: What ``predict`` returned; only the first prediction is read,
            because one call ran on one image.
        img_width: Width of the source image, in pixels.
        img_height: Height of the source image, in pixels.
        id2label: Class id to label. Taken off the model by the caller, which
            is what leaves this function testable without one.
        simplify: Whether to thin each polygon with Douglas-Peucker first.

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

    outlines = masks.xyn
    if len(boxes) != len(outlines):
        logger.warning(
            "Box and mask counts differ, pairing only what lines up",
            boxes=len(boxes),
            masks=len(outlines),
        )

    detections: list[Detection] = []
    dropped = 0
    # Boxes indexes but does not iterate, so pair the two by position.
    for i in range(min(len(boxes), len(outlines))):
        try:
            detection = _detection(
                boxes[i],
                outlines[i],
                img_width,
                img_height,
                id2label,
                simplify=simplify,
            )
        except Exception:
            # One unusable mask must not cost the page — but with no traceback a
            # silent drop is all anyone would ever see of it.
            dropped += 1
            logger.warning("Failed to process prediction", index=i, exc_info=True)
            continue

        if detection is None:
            dropped += 1
            continue
        detections.append(detection)

    if dropped:
        # A dropped marker silently re-files the rest of a book under the wrong
        # option, so say how many were lost rather than just that one was.
        logger.warning(
            "Dropped detections on this page", dropped=dropped, kept=len(detections)
        )

    return detections


class YOLO_SegmentationPredictor:
    """A YOLO segmentation checkpoint, read on first use.

    Answers to :class:`~digitex.pipeline.ports.RegionDetector`, which is the
    whole of what page extraction asks of it. Constructing one costs nothing —
    the checkpoint is opened on the first ``model`` access — so a caller can
    build a predictor while it is still deciding whether the run needs it.

    Args:
        model_path: Path to the trained YOLO checkpoint.
        simplify: Whether to run each mask through Douglas-Peucker. Label
            Studio pre-annotation asks for it, because an annotator drags those
            points by hand; extraction leaves it off and takes the mask as it
            came.
    """

    def __init__(
        self,
        model_path: str | Path,
        simplify: bool = False,
    ) -> None:
        self.model_path = model_path
        self.simplify = simplify
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO:
        """The checkpoint, loaded on first access and kept.

        Raises:
            RuntimeError: If the checkpoint cannot be loaded — a missing file,
                a corrupt pickle, or a torch that will not deserialize it.
        """
        if self._model is None:
            # Absolute, so which file is loaded does not depend on the
            # directory the process happened to start in.
            model_str = str(Path(self.model_path).resolve())
            try:
                with foreign_paths_readable():
                    self._model = YOLO(model_str, verbose=False)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model from {self.model_path}: {e}"
                ) from e
            logger.info("Model loaded successfully", model_path=self.model_path)

        return self._model

    def predict(self, image: Image.Image) -> list[Detection]:
        """Detect labelled regions on *image*, in source-image pixels.

        Args:
            image: PIL Image to predict on. Its size, not the size the model
                ran at, is what the returned polygons are scaled to.
        """
        img_width, img_height = image.size

        # ``end2end`` reaches the head through ``setup_model``, which YOLO runs
        # once per model instance — so it is read off the first predict() call
        # and every later one is ignored. Passing a constant is what keeps the
        # first call from deciding something else.
        #
        # ultralytics annotates `predict` for every mode it has at once: a
        # generator when `stream=True`, bare tensors for some heads. One PIL
        # image and no streaming is the list-of-Results case, which is the only
        # shape `detections_from` reads — so narrow it here, at the vendor
        # boundary, rather than widening the function to shapes it cannot take.
        preds = cast(
            "list[Results]",
            self.model.predict(
                image,
                conf=PREDICT_CONF,
                imgsz=PREDICT_IMGSZ,
                max_det=PREDICT_MAX_DET,
                end2end=PREDICT_END2END,
                agnostic_nms=PREDICT_AGNOSTIC_NMS,
                verbose=False,
            ),
        )
        return detections_from(
            preds,
            img_width,
            img_height,
            dict(self.model.names),
            simplify=self.simplify,
        )
