"""The Label Studio SDK, narrowed to what this project asks of the server.

Five calls: read a project's tasks, keep the ones nothing has answered yet,
write an annotation, delete a task, post predictions. Every one of them exists
because a command needs it — pre-annotating a project, repairing one whose
images moved, or retiring the pages an annotator skipped.

Deliberately not here: anything the SDK can already do that nothing asks for,
and anything that is not a request. Reading where a task's image lives is pure
URI parsing and belongs in :mod:`digitex.labeling.uris`; deciding what to do
with a task belongs to the caller that planned it.
"""

from typing import Any, Protocol

import structlog
from label_studio_sdk import LabelStudio

logger = structlog.get_logger()


class LabelStudioTask(Protocol):
    """The task attributes this package reads off the SDK's objects.

    Narrower than what the server sends, on purpose: a caller that needs a
    sixth field states it here, and the tests can then satisfy the whole
    protocol with a stub.
    """

    id: int
    data: dict[str, Any]
    is_labeled: bool
    # Both collections ride along on the listing because ``list_tasks`` asks
    # for fields="all". Asking the server per task instead — whether it holds
    # annotations, whether it holds a prediction — is one HTTP request for
    # every task in the project.
    annotations: list[dict[str, Any]]
    predictions: list[dict[str, Any]]


def _annotator_id(annotation: dict[str, Any]) -> Any:
    """Who made *annotation*, as the write side wants it.

    ``fields="all"`` expands ``completed_by`` into a user object while a plain
    listing leaves it an id, and the create endpoint takes the id.
    """
    annotator = annotation.get("completed_by")
    if isinstance(annotator, dict):
        return annotator["id"]
    return annotator


class LabelStudioClient:
    """Adapter over the Label Studio SDK.

    Args:
        url: Label Studio server URL.
        api_key: Label Studio API key.
    """

    def __init__(self, url: str, api_key: str) -> None:
        self._client = LabelStudio(base_url=url, api_key=api_key)

    def list_tasks(self, project_id: int) -> list[LabelStudioTask]:
        """Return every task in a project, annotations and predictions included.

        Args:
            project_id: Label Studio project ID.

        Returns:
            List of task objects.
        """
        tasks = list(self._client.tasks.list(project=project_id, fields="all"))
        logger.info("fetched_tasks", project_id=project_id, count=len(tasks))
        return tasks

    def get_unlabeled_tasks(self, project_id: int) -> list[LabelStudioTask]:
        """Return the tasks a prediction run may write to.

        A labeled task is a human's work, and a task that already carries a
        prediction would end up with a second guess stacked on the first.

        Args:
            project_id: Label Studio project ID.

        Returns:
            List of unlabeled task objects without predictions.
        """
        tasks = self.list_tasks(project_id)
        unlabeled = [
            task for task in tasks if not task.is_labeled and not task.predictions
        ]
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
        self._client.annotations.create(
            id=task_id,
            result=annotation.get("result", []),
            was_cancelled=bool(annotation.get("was_cancelled", False)),
            ground_truth=bool(annotation.get("ground_truth", False)),
            lead_time=annotation.get("lead_time"),
            completed_by=_annotator_id(annotation),
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
        # A run that predicted nothing would otherwise post an empty import.
        if not predictions:
            logger.warning("no_predictions", project_id=project_id)
            return

        self._client.projects.import_predictions(
            id=project_id,
            # The SDK declares Sequence[PredictionRequest] but coerces
            # mappings, and every caller here builds plain dicts.
            request=predictions,  # ty: ignore[invalid-argument-type]
        )
        logger.info(
            "uploaded_predictions",
            project_id=project_id,
            count=len(predictions),
        )
