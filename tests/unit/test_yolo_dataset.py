"""Tests for the YOLO DatasetCreator — ``create()`` is the test surface.

The creator's dependencies are a Label Studio export JSON and a directory of
images, both of which ``tmp_path`` stands in for, so every test drives the real
build and asserts on the returned :class:`Dataset` and the emitted tree.
"""

import json
from pathlib import Path

import pytest
import yaml

from digitex.ml.yolo.dataset import DatasetCreator

LABEL_LINE = "0 0.100000 0.200000 0.500000 0.200000 0.500000 0.800000 0.100000 0.800000"


def _annotation(image_name: str, label: str = "question") -> dict:
    return {
        "image": f"/data/local-files/?d={image_name}",
        "label": [
            {
                "polygonlabels": [label],
                "points": [[10.0, 20.0], [50.0, 20.0], [50.0, 80.0], [10.0, 80.0]],
            },
        ],
    }


def _creator(
    tmp_path: Path,
    annotations: list[dict],
    *,
    train_split: float = 0.8,
    images: tuple[str, ...] = (),
) -> DatasetCreator:
    """Build a creator rooted in tmp_path, seeding the export and image files."""
    annotations_file = tmp_path / "annotations.json"
    annotations_file.write_text(json.dumps(annotations))

    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    for name in images:
        (images_dir / name).write_bytes(b"fake image data")

    return DatasetCreator(
        annotations_file=annotations_file,
        images_dir=images_dir,
        dataset_dir=tmp_path / "dataset",
        train_split=train_split,
    )


def _all_files(dataset_dir: Path, suffix: str) -> set[str]:
    return {p.name for p in dataset_dir.rglob(f"*{suffix}")}


class TestCreate:
    def test_creates_all_three_splits_and_data_yaml(self, tmp_path: Path) -> None:
        names = tuple(f"image{i}.jpg" for i in range(10))
        creator = _creator(
            tmp_path,
            [_annotation(n) for n in names],
            train_split=0.7,
            images=names,
        )

        dataset = creator.create()

        assert dataset.dataset_dir == tmp_path / "dataset"
        for split in ("train", "val", "test"):
            assert (tmp_path / "dataset" / split).is_dir()
        assert (tmp_path / "dataset" / "data.yaml").exists()
        assert dataset.total == 10

    def test_split_counts_divide_the_remainder_sixty_forty(
        self, tmp_path: Path
    ) -> None:
        names = tuple(f"image{i}.jpg" for i in range(10))
        creator = _creator(
            tmp_path,
            [_annotation(n) for n in names],
            train_split=0.8,
            images=names,
        )

        dataset = creator.create()

        # 80% train; the remaining 20% splits 60/40 into val/test.
        assert (dataset.train, dataset.val, dataset.test) == (8, 1, 1)

    def test_every_annotated_image_is_copied_exactly_once(self, tmp_path: Path) -> None:
        names = tuple(f"image{i}.jpg" for i in range(6))
        creator = _creator(tmp_path, [_annotation(n) for n in names], images=names)

        creator.create()

        assert _all_files(tmp_path / "dataset", ".jpg") == set(names)

    def test_derives_a_sorted_class_map(self, tmp_path: Path) -> None:
        creator = _creator(
            tmp_path,
            [
                _annotation("image1.jpg", label="question"),
                _annotation("image2.jpg", label="option"),
            ],
            images=("image1.jpg", "image2.jpg"),
        )

        dataset = creator.create()

        assert dataset.classes == {0: "option", 1: "question"}

    def test_writes_a_yolo_label_file_beside_each_image(self, tmp_path: Path) -> None:
        creator = _creator(
            tmp_path,
            [_annotation("image0.jpg")],
            train_split=1.0,
            images=("image0.jpg",),
        )

        creator.create()

        label_path = tmp_path / "dataset" / "train" / "image0.txt"
        assert label_path.read_text() == LABEL_LINE

    def test_decodes_url_encoded_uris_to_filenames(self, tmp_path: Path) -> None:
        creator = _creator(
            tmp_path,
            [_annotation("images%5Cmy%20file.jpg")],
            train_split=1.0,
            images=("my file.jpg",),
        )

        dataset = creator.create()

        assert dataset.train == 1
        assert (tmp_path / "dataset" / "train" / "my file.jpg").exists()

    def test_reports_annotated_images_missing_from_disk(self, tmp_path: Path) -> None:
        creator = _creator(
            tmp_path,
            [_annotation("present.jpg"), _annotation("absent.jpg")],
            train_split=1.0,
            images=("present.jpg",),
        )

        dataset = creator.create()

        assert dataset.train == 1
        assert dataset.missing_images == ("absent.jpg",)
        assert not (tmp_path / "dataset" / "train" / "absent.jpg").exists()
        assert not (tmp_path / "dataset" / "train" / "absent.txt").exists()

    def test_skips_polygons_missing_a_label_or_points(self, tmp_path: Path) -> None:
        entry = {
            "image": "/data/local-files/?d=image.jpg",
            "label": [
                {"polygonlabels": [], "points": [[10.0, 20.0]]},
                {"polygonlabels": ["question"], "points": []},
            ],
        }
        creator = _creator(tmp_path, [entry], train_split=1.0, images=("image.jpg",))

        dataset = creator.create()

        assert dataset.classes == {0: "question"}
        # The unlabelled polygon is dropped; the point-less one keeps its class.
        assert (tmp_path / "dataset" / "train" / "image.txt").read_text() == "0 "

    def test_empty_export_still_produces_the_tree(self, tmp_path: Path) -> None:
        dataset = _creator(tmp_path, []).create()

        assert dataset.total == 0
        assert dataset.classes == {}
        assert dataset.missing_images == ()
        assert (tmp_path / "dataset" / "data.yaml").exists()

    def test_data_yaml_names_the_class_map_and_split_dirs(self, tmp_path: Path) -> None:
        creator = _creator(
            tmp_path,
            [
                _annotation("image1.jpg", label="question"),
                _annotation("image2.jpg", label="option"),
            ],
            images=("image1.jpg", "image2.jpg"),
        )

        creator.create()

        data = yaml.safe_load((tmp_path / "dataset" / "data.yaml").read_text())
        assert data["names"] == {0: "option", 1: "question"}
        assert (data["train"], data["val"], data["test"]) == ("train", "val", "test")

    def test_data_yaml_path_is_absolute_when_outside_cwd(self, tmp_path: Path) -> None:
        """tmp_path sits outside cwd, so ``path`` must not be a failed relative_to."""
        _creator(tmp_path, []).create()

        data = yaml.safe_load((tmp_path / "dataset" / "data.yaml").read_text())
        assert Path(data["path"]) == tmp_path / "dataset"


class TestDatasetValue:
    def test_total_sums_the_splits(self, tmp_path: Path) -> None:
        names = tuple(f"image{i}.jpg" for i in range(5))
        dataset = _creator(
            tmp_path,
            [_annotation(n) for n in names],
            train_split=0.6,
            images=names,
        ).create()

        assert dataset.total == dataset.train + dataset.val + dataset.test
        assert dataset.total == 5

    def test_is_immutable(self, tmp_path: Path) -> None:
        dataset = _creator(tmp_path, []).create()

        with pytest.raises(AttributeError):
            dataset.train = 99  # type: ignore[misc]
