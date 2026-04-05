"""YOLO dataset creation from Label Studio annotations."""

import json
import random
import shutil
import urllib.parse
from pathlib import Path

import structlog
import yaml

logger = structlog.get_logger()


class DatasetCreator:
    """Creates a YOLO dataset from Label Studio annotations.

    Args:
        annotations_file: Path to Label Studio export JSON.
        images_dir: Directory containing source images.
        dataset_dir: Output directory for train/val/test splits.
        train_split: Fraction of data for training (default 0.8).
    """

    def __init__(
        self,
        annotations_file: Path,
        images_dir: Path,
        dataset_dir: Path,
        train_split: float = 0.8,
    ) -> None:
        self.annotations_file = annotations_file
        self.images_dir = images_dir
        self.dataset_dir = dataset_dir
        self.classes: dict[int, str] = {}
        self.label2id: dict[str, int] = {}

        self.train_split = train_split
        self.val_split = 0.6 * (1 - self.train_split)
        self.test_split = 1 - self.train_split - self.val_split

        self._train_dir: Path | None = None
        self._val_dir: Path | None = None
        self._test_dir: Path | None = None

    @property
    def train_dir(self) -> Path:
        if self._train_dir is None:
            self._train_dir = self.dataset_dir / "train"
            self._train_dir.mkdir(parents=True, exist_ok=True)
        return self._train_dir

    @property
    def val_dir(self) -> Path:
        if self._val_dir is None:
            self._val_dir = self.dataset_dir / "val"
            self._val_dir.mkdir(parents=True, exist_ok=True)
        return self._val_dir

    @property
    def test_dir(self) -> Path:
        if self._test_dir is None:
            self._test_dir = self.dataset_dir / "test"
            self._test_dir.mkdir(parents=True, exist_ok=True)
        return self._test_dir

    @staticmethod
    def _extract_filename(image_uri: str) -> str:
        """Extract filename from Label Studio image URI.

        Args:
            image_uri: URI like /data/local-files/?d=training%5Cdata%5Cimages%5Cfile.jpg

        Returns:
            Filename (e.g. biology_2008_12_old.jpg).
        """
        parsed = urllib.parse.urlparse(image_uri)
        params = urllib.parse.parse_qs(parsed.query)
        path = urllib.parse.unquote(params.get("d", [""])[0])
        return Path(path).name

    @staticmethod
    def _parse_annotation(entry: dict, label2id: dict[str, int]) -> tuple[str, str]:
        """Parse a single Label Studio annotation entry into YOLO format.

        Args:
            entry: Annotation dict with 'image' and 'label' keys.
            label2id: Mapping from label name to class ID.

        Returns:
            Tuple of (filename, yolo_label_string).
        """
        filename = DatasetCreator._extract_filename(entry["image"])
        lines = []

        for polygon in entry.get("label", []):
            try:
                label_name = polygon["polygonlabels"][0]
                class_id = label2id[label_name]
                points = polygon["points"]
                coords = " ".join(f"{x / 100:.6f} {y / 100:.6f}" for x, y in points)
                lines.append(f"{class_id} {coords}")
            except (KeyError, IndexError) as exc:
                logger.warning("skipped_polygon", reason=str(exc), polygon=polygon)
                continue

        return filename, "\n".join(lines)

    def _load_annotations(self, shuffle: bool = True) -> dict[str, str]:
        """Load annotations.json, derive classes, and build image-to-label mapping.

        Args:
            shuffle: Whether to shuffle the result dict.

        Returns:
            Dict mapping image filename to YOLO label string.
        """
        with self.annotations_file.open("r", encoding="utf-8") as f:
            annotations = json.load(f)

        label_names: set[str] = set()
        for entry in annotations:
            for polygon in entry.get("label", []):
                for name in polygon.get("polygonlabels", []):
                    label_names.add(name)

        self.classes = {i: name for i, name in enumerate(sorted(label_names))}
        self.label2id = {v: k for k, v in self.classes.items()}
        logger.info("classes_derived", classes=self.classes)

        images_labels: dict[str, str] = {}
        for entry in annotations:
            filename, label_str = self._parse_annotation(entry, self.label2id)
            images_labels[filename] = label_str

        if shuffle:
            keys = list(images_labels.keys())
            random.shuffle(keys)
            images_labels = {k: images_labels[k] for k in keys}

        logger.info("loaded_annotations", count=len(images_labels))
        return images_labels

    def _write_label(self, label_str: str, dest_dir: Path, filename: str) -> None:
        """Write a YOLO label .txt file next to the image.

        Args:
            label_str: YOLO-formatted label content.
            dest_dir: Target directory (train/val/test).
            filename: Image filename (used to derive label filename).
        """
        label_path = dest_dir / (Path(filename).stem + ".txt")
        label_path.write_text(label_str, encoding="utf-8")

    def _copy_split(
        self,
        data: dict[str, str],
        dest_dir: Path,
    ) -> None:
        """Copy images and write labels for a data split.

        Args:
            data: Dict mapping image filename to YOLO label string.
            dest_dir: Target directory for this split.
        """
        for image_name, label_str in data.items():
            src = self.images_dir / image_name
            if not src.exists():
                logger.warning("image_not_found", name=image_name)
                continue

            shutil.copyfile(src, dest_dir / image_name)
            if label_str:
                self._write_label(label_str, dest_dir, image_name)

    def partition_data(self) -> None:
        """Split data into train/val/test and copy files."""
        data = self._load_annotations()

        num_train = int(len(data) * self.train_split)
        num_val = int(len(data) * self.val_split)

        keys = list(data.keys())
        train_data = {k: data[k] for k in keys[:num_train]}
        val_data = {k: data[k] for k in keys[num_train : num_train + num_val]}
        test_data = {k: data[k] for k in keys[num_train + num_val :]}

        self._copy_split(train_data, self.train_dir)
        self._copy_split(val_data, self.val_dir)
        self._copy_split(test_data, self.test_dir)

        logger.info(
            "partitioned",
            train=len(train_data),
            val=len(val_data),
            test=len(test_data),
        )

    def write_data_yaml(self) -> None:
        """Write data.yaml for YOLO training."""
        data = {
            "path": str(self.dataset_dir.resolve()),
            "train": "train",
            "val": "val",
            "test": "test",
            "names": self.classes,
        }

        yaml_path = self.dataset_dir / "data.yaml"
        yaml_path.write_text(
            yaml.dump(data, default_flow_style=False), encoding="utf-8"
        )

    def create(self) -> None:
        """Create the full YOLO dataset."""
        self.partition_data()
        self.write_data_yaml()
        logger.info("dataset_created", dir=str(self.dataset_dir))
