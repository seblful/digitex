"""Label Studio uploader for YOLO polygon annotations."""

from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import url2pathname

import structlog
from label_studio_sdk import LabelStudio
from tqdm import tqdm

logger = structlog.get_logger()


class LabelStudioUploader:
    """Uploads YOLO polygon labels to Label Studio as pre-annotations.

    Args:
        classes: Mapping from class ID to label name.
        url: Label Studio server URL.
        api_key: Label Studio API key.
    """

    def __init__(
        self,
        classes: dict[int, str],
        url: str,
        api_key: str,
    ) -> None:
        self._classes = classes
        self._client = LabelStudio(base_url=url, api_key=api_key)

    @staticmethod
    def _extract_filename(image_uri: str) -> str:
        """Extract filename from Label Studio image URI.

        Handles both formats:
          - /data/local-files/?file=data/page/images/name.jpg
          - /data/local-files/?d=Users%5C...%5Cname.jpg

        Args:
            image_uri: Image URI from task data.

        Returns:
            Image filename.
        """
        parsed = urlparse(image_uri)
        params = parse_qs(parsed.query)

        for key in ("file", "d"):
            if key in params:
                raw_path = url2pathname(params[key][0])
                return Path(raw_path).name

        return Path(image_uri).name

    def _get_task_map(self, project_id: int) -> dict[str, int]:
        """Build mapping from image filename to task ID.

        Args:
            project_id: Label Studio project ID.

        Returns:
            Dict mapping image filename to task ID.
        """
        task_map = {}
        tasks = self._client.tasks.list(project=project_id, fields="all")

        for task in tasks:
            image_uri = task.data.get("image", "")
            filename = self._extract_filename(image_uri)
            task_map[filename] = task.id

        logger.info("loaded_tasks", count=len(task_map))
        return task_map

    def parse_label(self, label_path: Path) -> list[dict]:
        """Parse YOLO polygon label file into Label Studio result dicts.

        Args:
            label_path: Path to YOLO .txt label file.

        Returns:
            List of annotation result dicts.
        """
        results = []

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue

                class_id = int(parts[0])
                coords = [float(x) * 100 for x in parts[1:]]
                points = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]

                results.append(
                    {
                        "from_name": "label",
                        "to_name": "image",
                        "type": "polygonlabels",
                        "value": {
                            "points": points,
                            "polygonlabels": [self._classes[class_id]],
                        },
                    }
                )

        return results

    def upload(
        self,
        project_id: int,
        labels_dir: Path,
        batch_size: int = 50,
    ) -> int:
        """Upload labels as predictions to existing tasks in Label Studio.

        Args:
            project_id: Label Studio project ID.
            labels_dir: Directory containing .txt label files.
            batch_size: Number of predictions per API call.

        Returns:
            Number of predictions uploaded.
        """
        task_map = self._get_task_map(project_id)
        label_files = sorted(labels_dir.glob("*.txt"))
        logger.info("found_label_files", count=len(label_files))

        predictions = []
        skipped = 0

        for label_path in tqdm(label_files, desc="Preparing predictions"):
            ls_image_name = label_path.stem + ".jpg"
            task_id = task_map.get(ls_image_name)

            if task_id is None:
                logger.warning("task_not_found", name=ls_image_name)
                skipped += 1
                continue

            results = self.parse_label(label_path)
            if not results:
                skipped += 1
                continue

            predictions.append(
                {
                    "task": task_id,
                    "result": results,
                    "model_version": "yolo-import",
                }
            )

        logger.info("predictions_prepared", total=len(predictions), skipped=skipped)

        for i in tqdm(range(0, len(predictions), batch_size), desc="Uploading"):
            batch = predictions[i : i + batch_size]
            self._client.projects.import_predictions(
                id=project_id,
                request=batch,
            )

        logger.info("upload_complete", total=len(predictions))
        return len(predictions)
