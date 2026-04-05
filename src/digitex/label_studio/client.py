"""Generic Label Studio SDK wrapper."""

from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import url2pathname

import structlog
from label_studio_sdk import LabelStudio

logger = structlog.get_logger()


class LabelStudioClient:
    """Generic wrapper around the Label Studio SDK.

    Args:
        url: Label Studio server URL.
        api_key: Label Studio API key.
    """

    def __init__(self, url: str, api_key: str) -> None:
        self._client = LabelStudio(base_url=url, api_key=api_key)

    def get_tasks(self, project_id: int) -> list:
        """Return all tasks for a project.

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
        tasks = self.get_tasks(project_id)
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

    @staticmethod
    def get_local_path(task) -> Path | None:
        """Extract filesystem path from a local-files URI in task data.

        Handles URIs of the form /data/local-files/?d=... or /data/local-files/?file=...

        Args:
            task: Label Studio task object.

        Returns:
            Path to the local file, or None if no valid URI is found.
        """
        image_uri = task.data.get("image", "")
        if not image_uri:
            return None

        parsed = urlparse(image_uri)
        params = parse_qs(parsed.query)

        for key in ("file", "d"):
            if key in params:
                raw_path = url2pathname(params[key][0])
                return Path(raw_path)

        return None

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

    def get_label_config(self, project_id: int) -> str:
        """Return the project's label config XML.

        Args:
            project_id: Label Studio project ID.

        Returns:
            Label config XML string.
        """
        project = self._client.projects.get(id=project_id)
        config = str(project.label_config)
        logger.info("fetched_label_config", project_id=project_id)
        return config

    def cancel_task(self, task_id: int) -> None:
        """Mark a task as cancelled.

        Args:
            task_id: Label Studio task ID.
        """
        self._client.tasks.update(
            id=str(task_id),
            meta={"is_cancelled": True},
        )
        logger.info("cancelled_task", task_id=task_id)
