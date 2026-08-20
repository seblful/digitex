"""Pre-annotating a project: the model's guess, uploaded as a prediction.

One run over the tasks nobody has answered yet. Every task is a page on this
machine's disk, and any of them can turn out not to be — the server holds tasks
for images that were moved, deleted, or never local at all — so a task the
model cannot read is logged and passed over rather than ending the run.

Deliberately not here: what the results mean. The prediction dicts this builds
go straight to the server, and correcting them is an annotator's job.
"""

from statistics import fmean
from typing import Any

import structlog
from PIL import Image

from digitex.domain.entities import Detection
from digitex.domain.geometry import pixel_to_percent
from digitex.labeling.client import LabelStudioClient, LabelStudioTask
from digitex.labeling.uris import task_image_path
from digitex.ml.predictors import YOLO_SegmentationPredictor

logger = structlog.get_logger()


class TaskPredictor:
    """Runs a segmentation model over a project's unannotated tasks.

    Both collaborators come in through the constructor — the same shape as
    :class:`~digitex.pipeline.page.PageExtractor` — so a test hands in fakes
    instead of patching module globals.

    Args:
        predictor: Segmentation model whose detections are uploaded.
        client: Label Studio API adapter.
        model_version: Version tag stamped on each uploaded prediction.
    """

    def __init__(
        self,
        predictor: YOLO_SegmentationPredictor,
        client: LabelStudioClient,
        model_version: str,
    ) -> None:
        self._predictor = predictor
        self._client = client
        self._model_version = model_version

    def _to_ls_results(
        self,
        detections: list[Detection],
        img_width: int,
        img_height: int,
    ) -> list[dict[str, Any]]:
        """Convert detections to Label Studio result format.

        The points are percentages, so reading them back needs no image size.
        The size is sent anyway because it is what Label Studio writes on its
        own exports, and a round trip that drops it stops looking like one.

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
                "original_width": img_width,
                "original_height": img_height,
                "score": detection.score,
                "value": {
                    "points": pixel_to_percent(
                        detection.polygon, img_width, img_height
                    ),
                    "polygonlabels": [detection.label],
                },
            }
            for detection in detections
        ]

    def _predict_task(self, task: LabelStudioTask) -> list[dict[str, Any]] | None:
        """Run the model over one task's page.

        Args:
            task: Label Studio task object.

        Returns:
            The Label Studio results for the page — empty when the model found
            nothing on it — or None when the page could not be predicted at
            all. Each way of failing is logged with the task it happened on,
            because a run over a thousand tasks is only debuggable per task.
        """
        image_path = task_image_path(task.data)
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

        img_width, img_height = image.size
        try:
            detections = self._predictor.predict(image)
        except Exception as e:
            logger.warning("skip_prediction_failed", task_id=task.id, error=str(e))
            return None

        return self._to_ls_results(detections, img_width, img_height)

    def _prediction(
        self, task: LabelStudioTask, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """One task's results, wrapped as the prediction the server imports."""
        return {
            "task": task.id,
            "result": results,
            "model_version": self._model_version,
            # One number per task is what the Data Manager sorts a review queue
            # by; the per-region scores stay in the results for anyone who wants
            # to know which region dragged it down. A page the model found
            # nothing on has no mean to take, and must not divide by zero.
            "score": fmean(result["score"] for result in results) if results else 0.0,
        }

    def predict_tasks(self, project_id: int) -> int:
        """Run predictions on all unannotated tasks in a project.

        Uploaded one task at a time rather than in one import at the end: a run
        over a large project is long enough to be interrupted, and what has been
        predicted so far should already be in the project when it is.

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
            # An empty list is a page with nothing on it, which is a prediction;
            # None is a page that could not be read, which is not.
            if results is None:
                continue
            try:
                self._client.upload_predictions(
                    project_id, [self._prediction(task, results)]
                )
            except Exception as e:
                logger.error("upload_failed", task_id=task.id, error=str(e))
                continue
            predicted += 1
            logger.info(
                "task_predicted",
                task_id=task.id,
                detections=len(results),
                progress=f"{predicted}/{len(tasks)}",
            )

        logger.info("predictions_complete", predicted=predicted, total=len(tasks))
        return predicted
