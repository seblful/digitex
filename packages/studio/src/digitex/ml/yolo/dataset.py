"""A YOLO train/val/test tree, built out of labelled images.

Takes :class:`~digitex.domain.annotations.AnnotatedImage` — a filename and
normalized polygons — copies each image next to the ``.txt`` label file YOLO
wants beside it, and writes the ``data.yaml`` that names the three splits.

Where the annotations came from is deliberately unknown here. This module used
to read Label Studio's export JSON itself, which made the model trainer a
consumer of the annotation tool's format while the layer contract said it knew
nothing about it. Parsing the export lives in :mod:`digitex.labeling.export`
now, and what arrives is already in nobody's format in particular.
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


@dataclass(frozen=True)
class Dataset:
    """What one dataset build produced.

    Returned by :meth:`DatasetCreator.create` so a caller can report on a build
    without reaching into the creator for its derived state. The three counts
    are images that were actually copied — an annotation whose image is not on
    disk lands in *missing_images* and in no split.
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

    def create(self) -> Dataset:
        """Build the dataset and return what landed on disk."""
        classes = self._class_map()
        labels = self._shuffled_labels(classes)

        copied: dict[str, int] = {}
        missing: list[str] = []
        for split, entries in self._partition(labels).items():
            copied[split], absent = self._copy_split(entries, self._split_dir(split))
            missing.extend(absent)

        self._write_data_yaml(classes)

        dataset = Dataset(
            dataset_dir=self._dataset_dir,
            classes=classes,
            train=copied["train"],
            val=copied["val"],
            test=copied["test"],
            missing_images=tuple(missing),
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

    def _class_map(self) -> dict[int, str]:
        """Class ids for the labels these annotations actually carry.

        Derived rather than configured, so a class nobody drew this round does
        not reserve an id.
        """
        drawn = {
            region.label for image in self._annotations for region in image.regions
        }
        classes = dict(enumerate(sorted(drawn)))
        logger.info("classes_derived", classes=classes)
        return classes

    @staticmethod
    def _label_text(image: AnnotatedImage, label2id: dict[str, int]) -> str:
        """One image's regions as the lines of its YOLO label file."""
        lines = []
        for region in image.regions:
            points = " ".join(f"{x:.6f} {y:.6f}" for x, y in region.polygon)
            lines.append(f"{label2id[region.label]} {points}")
        return "\n".join(lines)

    def _shuffled_labels(self, classes: dict[int, str]) -> dict[str, str]:
        """Every image's label-file text, in the order the splits are cut from."""
        label2id = {name: class_id for class_id, name in classes.items()}

        labels: dict[str, str] = {}
        for image in self._annotations:
            if image.filename in labels:
                # Overwriting is the old behaviour and stays; silently losing
                # the first one's polygons is what must not pass unremarked.
                logger.warning("duplicate_image_basename", name=image.filename)
            labels[image.filename] = self._label_text(image, label2id)

        order = list(labels)
        self._rng.shuffle(order)
        logger.info("loaded_annotations", count=len(order))
        return {name: labels[name] for name in order}

    def _partition(self, labels: dict[str, str]) -> dict[str, dict[str, str]]:
        """Cut the shuffled images into the three splits."""
        names = list(labels)
        train_end = int(len(names) * self._train_split)
        val_end = train_end + int(len(names) * self._val_split)
        return {
            "train": {name: labels[name] for name in names[:train_end]},
            "val": {name: labels[name] for name in names[train_end:val_end]},
            "test": {name: labels[name] for name in names[val_end:]},
        }

    def _split_dir(self, name: str) -> Path:
        """The directory a split writes into, made if it is not there yet."""
        path = self._dataset_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _copy_split(
        self, entries: dict[str, str], dest_dir: Path
    ) -> tuple[int, list[str]]:
        """Copy one split's images and label files into *dest_dir*.

        Returns:
            How many images were copied, and the annotated images that were not
            on disk to copy.
        """
        copied = 0
        missing: list[str] = []
        for image_name, label_text in entries.items():
            source = self._images_dir / image_name
            if not source.exists():
                logger.warning("image_not_found", name=image_name)
                missing.append(image_name)
                continue

            shutil.copyfile(source, dest_dir / image_name)
            if label_text:
                # A page an annotator drew nothing on is a negative example and
                # ships anyway, but a label file holding no lines is not a valid
                # YOLO instance — so it gets no file at all.
                label_path = dest_dir / f"{Path(image_name).stem}.txt"
                label_path.write_text(label_text, encoding="utf-8")
            copied += 1
        return copied, missing

    def _write_data_yaml(self, classes: dict[int, str]) -> None:
        """Write the ``data.yaml`` that points YOLO at the splits and classes."""
        data = {
            "path": self._yaml_base_path(),
            "train": "train",
            "val": "val",
            "test": "test",
            "names": classes,
        }
        yaml_path = self._dataset_dir / "data.yaml"
        yaml_path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")

    def _yaml_base_path(self) -> str:
        """YOLO resolves ``path`` against cwd; use absolute if we sit outside it."""
        try:
            return self._dataset_dir.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return self._dataset_dir.as_posix()
