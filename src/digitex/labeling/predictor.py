"""Prediction orchestrator for Label Studio tasks."""

from pathlib import Path

import structlog
from PIL import Image

from digitex.domain.entities import Detection
from digitex.domain.geometry import local_file_path, pixel_to_percent
from digitex.labeling.client import LabelStudioClient, LabelStudioTask
from digitex.ml.predictors import YOLO_SegmentationPredictor

logger = structlog.get_logger()


class TaskPredictor:
    """Orchestrates YOLO predictions on unannotated Label Studio tasks.

    Connects YOLO_SegmentationPredictor and LabelStudioClient to iterate
    unannotated tasks, run inference, and upload predictions immediately.

    Args:
        model_path: Path to the trained YOLO model file.
        url: Label Studio server URL.
        api_key: Label Studio API key.
        model_version: Model version tag. Defaults to model file stem.
    """

    def __init__(
        self,
        model_path: str,
        url: str,
        api_key: str,
        model_version: str = "",
    ) -> None:
        self._predictor = YOLO_SegmentationPredictor(model_path, simplify=True)
        self._client = LabelStudioClient(url, api_key)
        self._model_version = model_version or Path(model_path).stem

    def _to_ls_results(
        self,
        detections: list[Detection],
        img_width: int,
        img_height: int,
    ) -> list[dict]:
        """Convert detections to Label Studio result format.

        Args:
            detections: Detections carrying pixel-coordinate polygons.
            img_width: Image width in pixels.
            img_height: Image height in pixels.

        Returns:
            List of Label Studio result dicts with polygon labels.
        """
        return [
            {
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels",
                "value": {
                    "points": pixel_to_percent(det.polygon, img_width, img_height),
                    "polygonlabels": [det.label],
                },
            }
            for det in detections
        ]

    def _predict_task(self, task: LabelStudioTask) -> list[dict] | None:
        """Run prediction on a single Label Studio task.

        Args:
            task: Label Studio task object.

        Returns:
            List of Label Studio result dicts, or None if skipped.
        """
        image_path = local_file_path(task.data.get("image", ""))
        if image_path is None:
            logger.warning("skip_no_path", task_id=task.id)
            return None
        if not image_path.exists():
            logger.warning("skip_file_missing", task_id=task.id, path=str(image_path))
            return None
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning("skip_image_open_failed", task_id=task.id, error=str(e))
            return None
        try:
            detections = self._predictor.predict(image)
        except Exception as e:
            logger.warning("skip_prediction_failed", task_id=task.id, error=str(e))
            return None
        img_width, img_height = image.size
        return self._to_ls_results(detections, img_width, img_height)

    def predict_tasks(self, project_id: int) -> int:
        """Run predictions on all unannotated tasks in a project.

        Args:
            project_id: Label Studio project ID.

        Returns:
            Number of tasks successfully predicted.
        """
        tasks = self._client.get_unlabeled_tasks(project_id)
        logger.info("starting_predictions", project_id=project_id, total=len(tasks))
        predicted = 0
        for task in tasks:
            results = self._predict_task(task)
            if results is None:
                continue
            prediction = {
                "task": task.id,
                "result": results,
                "model_version": self._model_version,
            }
            try:
                self._client.upload_predictions(project_id, [prediction])
                predicted += 1
                logger.info(
                    "task_predicted",
                    task_id=task.id,
                    detections=len(results),
                    progress=f"{predicted}/{len(tasks)}",
                )
            except Exception as e:
                logger.error("upload_failed", task_id=task.id, error=str(e))
        logger.info("predictions_complete", predicted=predicted, total=len(tasks))
        return predicted
