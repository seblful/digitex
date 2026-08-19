"""Label Studio SDK adapter — the two calls the prediction run needs."""

from typing import Any, Protocol

import structlog
from label_studio_sdk import LabelStudio

logger = structlog.get_logger()


class LabelStudioTask(Protocol):
    """The task attributes this package reads off the SDK's objects."""

    id: int
    data: dict[str, Any]
    is_labeled: bool
    # ``list_tasks`` asks for fields="all", so annotations come back with the
    # task; ``labeling.repair`` and ``labeling.skipped`` both read them.
    annotations: list[dict[str, Any]]
    # Same listing, same reason: asking the server once per task whether it
    # already holds a prediction is a request per task in the project.
    predictions: list[dict[str, Any]]


class LabelStudioClient:
    """Adapter over the Label Studio SDK.

    Deliberately narrow: listing tasks, filtering them down to the unlabeled
    ones, and uploading predictions are the only operations callers perform, so
    they are the whole seam. Reading a task's local image path is not here —
    that is pure URI parsing and lives in
    :mod:`digitex.domain.geometry`.

    Args:
        url: Label Studio server URL.
        api_key: Label Studio API key.
    """

    def __init__(self, url: str, api_key: str) -> None:
        self._client = LabelStudio(base_url=url, api_key=api_key)

    def list_tasks(self, project_id: int) -> list[LabelStudioTask]:
        """Return every task in a project, annotations included.

        Args:
            project_id: Label Studio project ID.

        Returns:
            List of task objects.
        """
        tasks = list(self._client.tasks.list(project=project_id, fields="all"))
        logger.info("fetched_tasks", project_id=project_id, count=len(tasks))
        return tasks

    def get_unlabeled_tasks(self, project_id: int) -> list[LabelStudioTask]:
        """Return tasks where is_labeled is False and have no predictions.

        Args:
            project_id: Label Studio project ID.

        Returns:
            List of unlabeled task objects without predictions.
        """
        tasks = self.list_tasks(project_id)
        unlabeled = [t for t in tasks if not t.is_labeled and not t.predictions]
        logger.info(
            "filtered_unlabeled",
            project_id=project_id,
            total=len(tasks),
            unlabeled=len(unlabeled),
        )
        return unlabeled

    def create_annotation(self, task_id: int, annotation: dict[str, Any]) -> None:
        """Recreate one annotation, as read off another task, on this one.

        Only what an annotator produced crosses over — the result, whether they
        skipped the task, how long it took, and who they were. Identity and
        timestamps are the server's to assign, so a moved annotation is a new
        record of the same work.

        Args:
            task_id: Label Studio task the annotation is created on.
            annotation: An annotation as ``list_tasks`` returns it.
        """
        # fields="all" expands the annotator into an object; the write side
        # takes the id.
        annotator = annotation.get("completed_by")
        self._client.annotations.create(
            id=task_id,
            result=annotation.get("result", []),
            was_cancelled=bool(annotation.get("was_cancelled", False)),
            ground_truth=bool(annotation.get("ground_truth", False)),
            lead_time=annotation.get("lead_time"),
            completed_by=annotator["id"] if isinstance(annotator, dict) else annotator,
        )
        logger.info("created_annotation", task_id=task_id)

    def delete_task(self, task_id: int) -> None:
        """Delete a task, and with it everything the server hangs off one."""
        # The SDK types the id as a string and formats it into the URL.
        self._client.tasks.delete(id=str(task_id))
        logger.info("deleted_task", task_id=task_id)

    def upload_predictions(
        self, project_id: int, predictions: list[dict[str, Any]]
    ) -> None:
        """Upload predictions to a project.

        Each prediction carries its own ``model_version`` key; the SDK reads it
        off the payload, so there is no separate tag to pass here.

        Args:
            project_id: Label Studio project ID.
            predictions: List of prediction dicts.
        """
        if not predictions:
            logger.warning("no_predictions", project_id=project_id)
            return

        self._client.projects.import_predictions(
            # The SDK declares Sequence[PredictionRequest] but coerces mappings,
            # and every caller here builds plain dicts.
            id=project_id,
            request=predictions,  # ty: ignore[invalid-argument-type]
        )
        logger.info(
            "uploaded_predictions",
            project_id=project_id,
            count=len(predictions),
        )
