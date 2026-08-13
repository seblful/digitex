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
    # task; the cancelled-task sweep in training/scripts reads them.
    annotations: list[dict[str, Any]]


class LabelStudioClient:
    """Adapter over the Label Studio SDK.

    Deliberately narrow: listing tasks, filtering them down to the unlabeled
    ones, and uploading predictions are the only operations callers perform, so
    they are the whole seam. Reading a task's local image path is not here —
    that is pure URI parsing and lives in
    :mod:`digitex.label_studio.geometry`.

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
        unlabeled = []
        for t in tasks:
            if t.is_labeled:
                continue
            predictions = list(self._client.predictions.list(task=t.id))
            if predictions:
                continue
            unlabeled.append(t)
        logger.info(
            "filtered_unlabeled",
            project_id=project_id,
            total=len(tasks),
            unlabeled=len(unlabeled),
        )
        return unlabeled

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
