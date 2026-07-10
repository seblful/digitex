"""Tests for the YOLO DatasetCreator (Label Studio export → YOLO dataset)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from digitex.ml.yolo.dataset import DatasetCreator


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
    annotations: list[dict] | None = None,
    train_split: float = 0.8,
    image_count: int = 0,
) -> DatasetCreator:
    """Build a DatasetCreator rooted in tmp_path, optionally seeding files."""
    annotations_file = tmp_path / "annotations.json"
    images_dir = tmp_path / "images"
    if annotations is not None:
        annotations_file.write_text(json.dumps(annotations))
    if image_count:
        images_dir.mkdir(exist_ok=True)
        for i in range(image_count):
            (images_dir / f"image{i}.jpg").write_bytes(b"fake")
    return DatasetCreator(
        annotations_file=annotations_file,
        images_dir=images_dir,
        dataset_dir=tmp_path / "dataset",
        train_split=train_split,
    )


class TestDatasetCreatorInit:
    def test_init_stores_paths_and_derives_splits(self, tmp_path: Path) -> None:
        creator = _creator(tmp_path, train_split=0.8)

        assert creator.annotations_file == tmp_path / "annotations.json"
        assert creator.images_dir == tmp_path / "images"
        assert creator.dataset_dir == tmp_path / "dataset"
        assert creator.train_split == 0.8
        assert creator.val_split == pytest.approx(0.12, rel=1e-9)
        assert creator.test_split == pytest.approx(0.08, rel=1e-9)

    @pytest.mark.parametrize(
        ("split", "expected_dir"),
        [("train", "train"), ("val", "val"), ("test", "test")],
        ids=["train", "val", "test"],
    )
    def test_split_dir_properties_create_directories(
        self, tmp_path: Path, split: str, expected_dir: str
    ) -> None:
        creator = _creator(tmp_path)

        split_dir = getattr(creator, f"{split}_dir")

        assert split_dir == tmp_path / "dataset" / expected_dir
        assert split_dir.exists()


class TestDatasetCreatorAnnotations:
    def test_extract_filename_parses_label_studio_uri(self) -> None:
        uri = "/data/local-files/?d=training%5Cdata%5Cimages%5Cbiology_2008_12.jpg"
        assert DatasetCreator._extract_filename(uri) == "biology_2008_12.jpg"

    def test_extract_filename_decodes_spaces(self) -> None:
        uri = "/data/local-files/?d=images%5Cfile%20name.jpg"
        assert DatasetCreator._extract_filename(uri) == "file name.jpg"

    def test_parse_annotation_converts_to_yolo_format(self) -> None:
        entry = _annotation("image.jpg")
        label2id = {"question": 0, "option": 1}

        filename, label_str = DatasetCreator._parse_annotation(entry, label2id)

        assert filename == "image.jpg"
        assert (
            "0 0.100000 0.200000 0.500000 0.200000 0.500000 0.800000 0.100000 0.800000"
            in label_str
        )

    def test_parse_annotation_skips_polygons_missing_keys(self) -> None:
        entry = {
            "image": "/data/local-files/?d=image.jpg",
            "label": [
                {"polygonlabels": [], "points": [[10.0, 20.0]]},
                {"polygonlabels": ["question"], "points": []},
            ],
        }

        filename, label_str = DatasetCreator._parse_annotation(entry, {"question": 0})

        assert filename == "image.jpg"
        assert label_str == "0 "

    def test_load_annotations_derives_sorted_classes(self, tmp_path: Path) -> None:
        creator = _creator(
            tmp_path,
            annotations=[
                _annotation("image1.jpg", label="question"),
                _annotation("image2.jpg", label="option"),
            ],
        )

        images_labels = creator._load_annotations(shuffle=False)

        assert len(images_labels) == 2
        assert "image1.jpg" in images_labels
        assert "image2.jpg" in images_labels
        assert creator.classes == {0: "option", 1: "question"}
        assert creator.label2id == {"option": 0, "question": 1}

    def test_load_annotations_empty(self, tmp_path: Path) -> None:
        creator = _creator(tmp_path, annotations=[])
        assert creator._load_annotations(shuffle=False) == {}


class TestDatasetCreatorFiles:
    def test_write_label_creates_txt_next_to_image_name(self, tmp_path: Path) -> None:
        creator = _creator(tmp_path)
        label_str = "0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8"
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        creator._write_label(label_str, dest_dir, "image.jpg")

        label_path = dest_dir / "image.txt"
        assert label_path.exists()
        assert label_path.read_text() == label_str

    def test_copy_split_copies_images_and_writes_labels(self, tmp_path: Path) -> None:
        creator = _creator(tmp_path, image_count=1)
        (tmp_path / "images" / "image1.jpg").write_bytes(b"fake image data")
        data = {"image1.jpg": "0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8"}
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        creator._copy_split(data, dest_dir)

        assert (dest_dir / "image1.jpg").exists()
        assert (
            dest_dir / "image1.txt"
        ).read_text() == "0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8"

    def test_copy_split_skips_missing_images(self, tmp_path: Path) -> None:
        creator = _creator(tmp_path)
        data = {"missing.jpg": "0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8"}
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        creator._copy_split(data, dest_dir)

        assert not (dest_dir / "missing.jpg").exists()
        assert not (dest_dir / "missing.txt").exists()


class TestDatasetCreatorPipeline:
    def test_partition_data_creates_all_splits(self, tmp_path: Path) -> None:
        creator = _creator(
            tmp_path,
            annotations=[_annotation(f"image{i}.jpg") for i in range(10)],
            train_split=0.7,
            image_count=10,
        )

        creator.partition_data()

        dataset_dir = tmp_path / "dataset"
        assert (dataset_dir / "train").exists()
        assert (dataset_dir / "val").exists()
        assert (dataset_dir / "test").exists()

    def test_write_data_yaml(self, tmp_path: Path) -> None:
        creator = _creator(tmp_path)
        (tmp_path / "dataset").mkdir()
        creator.classes = {0: "question", 1: "option"}

        with patch("digitex.ml.yolo.dataset.Path.cwd", return_value=tmp_path):
            creator.write_data_yaml()

        yaml_path = tmp_path / "dataset" / "data.yaml"
        assert yaml_path.exists()
        with yaml_path.open() as f:
            data = yaml.safe_load(f)
        assert data["names"] == {0: "question", 1: "option"}
        assert data["train"] == "train"
        assert data["val"] == "val"
        assert data["test"] == "test"

    def test_create_runs_full_pipeline(self, tmp_path: Path) -> None:
        creator = _creator(
            tmp_path,
            annotations=[_annotation(f"image{i}.jpg") for i in range(5)],
            train_split=0.6,
            image_count=5,
        )

        with patch("digitex.ml.yolo.dataset.Path.cwd", return_value=tmp_path):
            creator.create()

        dataset_dir = tmp_path / "dataset"
        assert (dataset_dir / "train").exists()
        assert (dataset_dir / "val").exists()
        assert (dataset_dir / "test").exists()
        assert (dataset_dir / "data.yaml").exists()
