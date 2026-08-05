"""YOLO dataset creation from Label Studio annotations."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import structlog
import yaml

from digitex.label_studio.geometry import local_file_path, percent_to_normalized

logger = structlog.get_logger()

_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Dataset:
    """What one dataset build produced.

    Returned by :meth:`DatasetCreator.create` so callers can report on a build
    without reaching into the creator for its derived state. The split counts
    are images actually copied — an annotation whose image is missing from disk
    lands in *missing_images* instead.
    """

    dataset_dir: Path
    classes: dict[int, str]
    train: int
    val: int
    test: int
    missing_images: tuple[str, ...]

    @property
    def total(self) -> int:
        """Images copied across all three splits."""
        return self.train + self.val + self.test


class DatasetCreator:
    """Builds a YOLO train/val/test dataset from a Label Studio export.

    ``create`` is the whole interface: it reads the export, derives the class
    map, shuffles and splits the images, copies each one with its YOLO label
    file, writes ``data.yaml``, and returns a :class:`Dataset` describing the
    result.

    Args:
        annotations_file: Path to the Label Studio export JSON.
        images_dir: Directory holding the source images.
        dataset_dir: Output directory for the train/val/test splits.
        train_split: Fraction of images used for training. The remainder is
            divided 60/40 between val and test.
    """

    def __init__(
        self,
        annotations_file: Path,
        images_dir: Path,
        dataset_dir: Path,
        train_split: float = 0.8,
    ) -> None:
        self._annotations_file = annotations_file
        self._images_dir = images_dir
        self._dataset_dir = dataset_dir
        self._train_split = train_split
        self._val_split = 0.6 * (1 - train_split)
        self._classes: dict[int, str] = {}
        self._missing: list[str] = []

    def create(self) -> Dataset:
        """Build the dataset and return what landed on disk."""
        labels = self._load_annotations()
        splits = self._partition(labels)

        counts = [
            self._copy_split(data, self._split_dir(name))
            for name, data in zip(_SPLITS, splits, strict=True)
        ]
        self._write_data_yaml()

        dataset = Dataset(
            dataset_dir=self._dataset_dir,
            classes=dict(self._classes),
            train=counts[0],
            val=counts[1],
            test=counts[2],
            missing_images=tuple(self._missing),
        )
        logger.info(
            "dataset_created",
            dir=str(self._dataset_dir),
            train=dataset.train,
            val=dataset.val,
            test=dataset.test,
            missing=len(dataset.missing_images),
        )
        return dataset

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _extract_filename(image_uri: str) -> str | None:
        """Filename from a Label Studio local-files URI, or None if unparseable."""
        path = local_file_path(image_uri)
        return path.name if path else None

    @staticmethod
    def _parse_annotation(
        entry: dict, label2id: dict[str, int]
    ) -> tuple[str | None, str]:
        """One export entry to ``(filename, YOLO label text)``.

        Polygons missing a label or their points are skipped with a warning —
        a partially malformed export still yields a usable dataset.
        """
        filename = DatasetCreator._extract_filename(entry["image"])
        lines = []

        for polygon in entry.get("label", []):
            try:
                label_name = polygon["polygonlabels"][0]
                class_id = label2id[label_name]
                normalized = percent_to_normalized(polygon["points"])
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
                lines.append(f"{class_id} {coords}")
            except (KeyError, IndexError) as exc:
                logger.warning("skipped_polygon", reason=str(exc), polygon=polygon)
                continue

        return filename, "\n".join(lines)

    def _load_annotations(self) -> dict[str, str]:
        """Read the export, derive the class map, and shuffle the image order."""
        with self._annotations_file.open("r", encoding="utf-8") as f:
            annotations = json.load(f)

        label_names: set[str] = set()
        for entry in annotations:
            for polygon in entry.get("label", []):
                for name in polygon.get("polygonlabels", []):
                    label_names.add(name)

        self._classes = dict(enumerate(sorted(label_names)))
        label2id = {v: k for k, v in self._classes.items()}
        logger.info("classes_derived", classes=self._classes)

        images_labels: dict[str, str] = {}
        for entry in annotations:
            filename, label_str = self._parse_annotation(entry, label2id)
            if filename is None:
                logger.warning("skipped_entry_no_local_path", image=entry.get("image"))
                continue
            images_labels[filename] = label_str

        keys = list(images_labels)
        random.shuffle(keys)
        logger.info("loaded_annotations", count=len(keys))
        return {k: images_labels[k] for k in keys}

    def _partition(self, data: dict[str, str]) -> tuple[dict[str, str], ...]:
        keys = list(data)
        num_train = int(len(data) * self._train_split)
        cut = num_train + int(len(data) * self._val_split)
        return (
            {k: data[k] for k in keys[:num_train]},
            {k: data[k] for k in keys[num_train:cut]},
            {k: data[k] for k in keys[cut:]},
        )

    def _split_dir(self, name: str) -> Path:
        path = self._dataset_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _copy_split(self, data: dict[str, str], dest_dir: Path) -> int:
        """Copy a split's images plus their labels. Returns images copied."""
        copied = 0
        for image_name, label_str in data.items():
            src = self._images_dir / image_name
            if not src.exists():
                logger.warning("image_not_found", name=image_name)
                self._missing.append(image_name)
                continue

            shutil.copyfile(src, dest_dir / image_name)
            if label_str:
                label_path = dest_dir / (Path(image_name).stem + ".txt")
                label_path.write_text(label_str, encoding="utf-8")
            copied += 1
        return copied

    def _write_data_yaml(self) -> None:
        data = {
            "path": self._yaml_base_path(),
            "train": "train",
            "val": "val",
            "test": "test",
            "names": self._classes,
        }
        yaml_path = self._dataset_dir / "data.yaml"
        yaml_path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")

    def _yaml_base_path(self) -> str:
        """YOLO resolves ``path`` against cwd; use absolute if we sit outside it."""
        try:
            return self._dataset_dir.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return self._dataset_dir.as_posix()
