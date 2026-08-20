"""Turning labelled images into a YOLO train/val/test tree.

Takes :class:`~digitex.domain.annotations.AnnotatedImage` — a filename and
normalized polygons — and knows nothing about where they came from. It used to
read Label Studio's export JSON directly, which made the model trainer a
consumer of the annotation tool's format while the layer contract said it knew
nothing about it. Parsing the export lives in
:mod:`digitex.labeling.export` now.
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence

    from digitex.domain.annotations import AnnotatedImage

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
    """Builds a YOLO train/val/test dataset from labelled images.

    ``create`` is the whole interface: it derives the class map from what the
    annotations actually contain, shuffles and splits the images, copies each
    one with its YOLO label file, writes ``data.yaml``, and returns a
    :class:`Dataset` describing the result.

    Args:
        annotations: The labelled images to build from, in any order. Where two
            share a filename the later one wins, and says so — the export
            addresses images by URI and only the basename survives, so two
            annotation batches can each hold a ``30.jpg``.
        images_dir: Directory holding the source images.
        dataset_dir: Output directory for the train/val/test splits.
        train_split: Fraction of images used for training. The remainder is
            divided 60/40 between val and test.
        seed: Seeds the shuffle the splits are cut from. Fixed by default,
            because an unseeded one re-deals train/val/test on every build —
            and a model trained before the rebuild has then been trained on
            part of the test split it is about to be scored against. Matches
            the ``seed`` the train config hands YOLO.
    """

    def __init__(
        self,
        annotations: Sequence[AnnotatedImage],
        images_dir: Path,
        dataset_dir: Path,
        train_split: float = 0.8,
        seed: int = 0,
    ) -> None:
        self._annotations = annotations
        self._images_dir = images_dir
        self._dataset_dir = dataset_dir
        self._train_split = train_split
        self._val_split = 0.6 * (1 - train_split)
        self._rng = random.Random(seed)
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
    def _label_text(image: AnnotatedImage, label2id: dict[str, int]) -> str:
        """One image's regions as the lines of its YOLO label file."""
        lines = []
        for region in image.regions:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in region.polygon)
            lines.append(f"{label2id[region.label]} {coords}")
        return "\n".join(lines)

    def _load_annotations(self) -> dict[str, str]:
        """Derive the class map, then shuffle the image order.

        The class map comes from the labels the annotations actually carry, so
        a class nobody drew this round does not reserve an id — which is why it
        is derived here rather than configured.
        """
        label_names = {
            region.label for image in self._annotations for region in image.regions
        }
        self._classes = dict(enumerate(sorted(label_names)))
        label2id = {v: k for k, v in self._classes.items()}
        logger.info("classes_derived", classes=self._classes)

        images_labels: dict[str, str] = {}
        for image in self._annotations:
            if image.filename in images_labels:
                # Silently losing the first one's polygons must not pass
                # unremarked.
                logger.warning("duplicate_image_basename", name=image.filename)
            images_labels[image.filename] = self._label_text(image, label2id)

        keys = list(images_labels)
        self._rng.shuffle(keys)
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
