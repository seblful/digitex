"""Label Studio SDK adapter — the two calls the prediction run needs."""

import structlog
from label_studio_sdk import LabelStudio

logger = structlog.get_logger()


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

    def list_tasks(self, project_id: int) -> list:
        """Return every task in a project, annotations included.

        Args:
            project_id: Label Studio project ID.

        Returns:
            List of task objects.
        """
        tasks = list(self._client.tasks.list(project=project_id, fields="all"))
        logger.info("fetched_tasks", project_id=project_id, count=len(tasks))
        return tasks

    def get_unlabeled_tasks(self, project_id: int) -> list:
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
        self,
        project_id: int,
        predictions,
        model_version: str = "",
    ) -> None:
        """Upload predictions to a project.

        Args:
            project_id: Label Studio project ID.
            predictions: List of prediction dicts.
            model_version: Model version tag to attach.
        """
        if not predictions:
            logger.warning("no_predictions", project_id=project_id)
            return

        self._client.projects.import_predictions(
            id=project_id,
            request=predictions,  # type: ignore[arg-type]
        )
        logger.info(
            "uploaded_predictions",
            project_id=project_id,
            count=len(predictions),
            model_version=model_version,
        )
