"""Tests for the DatasetCreator class."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from digitex.ml.yolo.dataset import DatasetCreator


class TestDatasetCreator:
    """Test suite for DatasetCreator class."""

    def test_init(self, tmp_path: Path) -> None:
        """Test DatasetCreator initialization."""
        annotations_file = tmp_path / "annotations.json"
        images_dir = tmp_path / "images"
        dataset_dir = tmp_path / "dataset"

        creator = DatasetCreator(
            annotations_file=annotations_file,
            images_dir=images_dir,
            dataset_dir=dataset_dir,
            train_split=0.8,
        )

        assert creator.annotations_file == annotations_file
        assert creator.images_dir == images_dir
        assert creator.dataset_dir == dataset_dir
        assert creator.train_split == 0.8
        assert creator.val_split == pytest.approx(0.12, rel=1e-9)
        assert creator.test_split == pytest.approx(0.08, rel=1e-9)
        assert creator._train_dir is None
        assert creator._val_dir is None
        assert creator._test_dir is None

    def test_init_default_splits(self, tmp_path: Path) -> None:
        """Test DatasetCreator with default splits."""
        annotations_file = tmp_path / "annotations.json"
        images_dir = tmp_path / "images"
        dataset_dir = tmp_path / "dataset"

        creator = DatasetCreator(
            annotations_file=annotations_file,
            images_dir=images_dir,
            dataset_dir=dataset_dir,
        )

        assert creator.train_split == 0.8
        assert creator.val_split == pytest.approx(0.12, rel=1e-9)
        assert creator.test_split == pytest.approx(0.08, rel=1e-9)

    def test_train_dir_creates_directory(self, tmp_path: Path) -> None:
        """Test train_dir property creates directory on access."""
        annotations_file = tmp_path / "annotations.json"
        images_dir = tmp_path / "images"
        dataset_dir = tmp_path / "dataset"

        creator = DatasetCreator(
            annotations_file=annotations_file,
            images_dir=images_dir,
            dataset_dir=dataset_dir,
        )

        train_dir = creator.train_dir

        assert train_dir == dataset_dir / "train"
        assert train_dir.exists()

    def test_val_dir_creates_directory(self, tmp_path: Path) -> None:
        """Test val_dir property creates directory on access."""
        annotations_file = tmp_path / "annotations.json"
        images_dir = tmp_path / "images"
        dataset_dir = tmp_path / "dataset"

        creator = DatasetCreator(
            annotations_file=annotations_file,
            images_dir=images_dir,
            dataset_dir=dataset_dir,
        )

        val_dir = creator.val_dir

        assert val_dir == dataset_dir / "val"
        assert val_dir.exists()

    def test_test_dir_creates_directory(self, tmp_path: Path) -> None:
        """Test test_dir property creates directory on access."""
        annotations_file = tmp_path / "annotations.json"
        images_dir = tmp_path / "images"
        dataset_dir = tmp_path / "dataset"

        creator = DatasetCreator(
            annotations_file=annotations_file,
            images_dir=images_dir,
            dataset_dir=dataset_dir,
        )

        test_dir = creator.test_dir

        assert test_dir == dataset_dir / "test"
        assert test_dir.exists()

    def test_extract_filename(self) -> None:
        """Test _extract_filename parses Label Studio URI correctly."""
        uri = "/data/local-files/?d=training%5Cdata%5Cimages%5Cbiology_2008_12.jpg"
        filename = DatasetCreator._extract_filename(uri)
        assert filename == "biology_2008_12.jpg"

    def test_extract_filename_with_spaces(self) -> None:
        """Test _extract_filename handles URL-encoded spaces."""
        uri = "/data/local-files/?d=images%5Cfile%20name.jpg"
        filename = DatasetCreator._extract_filename(uri)
        assert filename == "file name.jpg"

    def test_parse_annotation(self) -> None:
        """Test _parse_annotation converts to YOLO format."""
        entry = {
            "image": "/data/local-files/?d=image.jpg",
            "label": [
                {
                    "polygonlabels": ["question"],
                    "points": [[10.0, 20.0], [50.0, 20.0], [50.0, 80.0], [10.0, 80.0]],
                },
            ],
        }
        label2id = {"question": 0, "option": 1}

        filename, label_str = DatasetCreator._parse_annotation(entry, label2id)

        assert filename == "image.jpg"
        assert (
            "0 0.100000 0.200000 0.500000 0.200000 0.500000 0.800000 0.100000 0.800000"
            in label_str
        )

    def test_parse_annotation_skips_invalid(self) -> None:
        """Test _parse_annotation skips polygons without required keys."""
        entry = {
            "image": "/data/local-files/?d=image.jpg",
            "label": [
                {"polygonlabels": [], "points": [[10.0, 20.0]]},
                {"polygonlabels": ["question"], "points": []},
            ],
        }
        label2id = {"question": 0}

        filename, label_str = DatasetCreator._parse_annotation(entry, label2id)

        assert filename == "image.jpg"
        assert label_str == "0 "

    def test_load_annotations(self, tmp_path: Path) -> None:
        """Test _load_annotations parses Label Studio export."""
        annotations_file = tmp_path / "annotations.json"
        images_dir = tmp_path / "images"
        dataset_dir = tmp_path / "dataset"

        annotations = [
            {
                "image": "/data/local-files/?d=image1.jpg",
                "label": [
                    {
                        "polygonlabels": ["question"],
                        "points": [
                            [10.0, 20.0],
                            [50.0, 20.0],
                            [50.0, 80.0],
                            [10.0, 80.0],
                        ],
                    },
                ],
            },
            {
                "image": "/data/local-files/?d=image2.jpg",
                "label": [
                    {
                        "polygonlabels": ["option"],
                        "points": [
                            [30.0, 40.0],
                            [70.0, 40.0],
                            [70.0, 90.0],
                            [30.0, 90.0],
                        ],
                    },
                ],
            },
        ]
        annotations_file.write_text(json.dumps(annotations))

        creator = DatasetCreator(
            annotations_file=annotations_file,
            images_dir=images_dir,
            dataset_dir=dataset_dir,
        )

        images_labels = creator._load_annotations(shuffle=False)

        assert len(images_labels) == 2
        assert "image1.jpg" in images_labels
        assert "image2.jpg" in images_labels
        assert creator.classes == {0: "option", 1: "question"}
        assert creator.label2id == {"option": 0, "question": 1}

    def test_load_annotations_empty(self, tmp_path: Path) -> None:
        """Test _load_annotations handles empty annotations."""
        annotations_file = tmp_path / "annotations.json"
        images_dir = tmp_path / "images"
        dataset_dir = tmp_path / "dataset"

        annotations_file.write_text("[]")

        creator = DatasetCreator(
            annotations_file=annotations_file,
            images_dir=images_dir,
            dataset_dir=dataset_dir,
        )

        images_labels = creator._load_annotations(shuffle=False)

        assert len(images_labels) == 0

    def test_write_label(self, tmp_path: Path) -> None:
        """Test _write_label creates label file."""
        annotations_file = tmp_path / "annotations.json"
        images_dir = tmp_path / "images"
        dataset_dir = tmp_path / "dataset"

        creator = DatasetCreator(
            annotations_file=annotations_file,
            images_dir=images_dir,
            dataset_dir=dataset_dir,
        )

        label_str = "0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8"
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        creator._write_label(label_str, dest_dir, "image.jpg")

        label_path = dest_dir / "image.txt"
        assert label_path.exists()
        assert label_path.read_text() == label_str

    def test_copy_split(self, tmp_path: Path) -> None:
        """Test _copy_split copies images and writes labels."""
        annotations_file = tmp_path / "annotations.json"
        images_dir = tmp_path / "images"
        dataset_dir = tmp_path / "dataset"

        images_dir.mkdir()
        source_image = images_dir / "image1.jpg"
        source_image.write_bytes(b"fake image data")

        creator = DatasetCreator(
            annotations_file=annotations_file,
            images_dir=images_dir,
            dataset_dir=dataset_dir,
        )

        data = {"image1.jpg": "0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8"}
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        creator._copy_split(data, dest_dir)

        assert (dest_dir / "image1.jpg").exists()
        assert (dest_dir / "image1.txt").exists()
        assert (
            dest_dir / "image1.txt"
        ).read_text() == "0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8"

    def test_copy_split_skips_missing_images(self, tmp_path: Path) -> None:
        """Test _copy_split skips images that don't exist."""
        annotations_file = tmp_path / "annotations.json"
        images_dir = tmp_path / "images"
        dataset_dir = tmp_path / "dataset"

        creator = DatasetCreator(
            annotations_file=annotations_file,
            images_dir=images_dir,
            dataset_dir=dataset_dir,
        )

        data = {"missing.jpg": "0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8"}
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        creator._copy_split(data, dest_dir)

        assert not (dest_dir / "missing.jpg").exists()
        assert not (dest_dir / "missing.txt").exists()

    def test_partition_data(self, tmp_path: Path) -> None:
        """Test partition_data splits data correctly."""
        annotations_file = tmp_path / "annotations.json"
        images_dir = tmp_path / "images"
        dataset_dir = tmp_path / "dataset"

        annotations = [
            {
                "image": f"/data/local-files/?d=image{i}.jpg",
                "label": [
                    {
                        "polygonlabels": ["question"],
                        "points": [
                            [10.0, 20.0],
                            [50.0, 20.0],
                            [50.0, 80.0],
                            [10.0, 80.0],
                        ],
                    },
                ],
            }
            for i in range(10)
        ]
        annotations_file.write_text(json.dumps(annotations))
        images_dir.mkdir()

        for i in range(10):
            (images_dir / f"image{i}.jpg").write_bytes(b"fake")

        creator = DatasetCreator(
            annotations_file=annotations_file,
            images_dir=images_dir,
            dataset_dir=dataset_dir,
            train_split=0.7,
        )

        creator.partition_data()

        assert (dataset_dir / "train").exists()
        assert (dataset_dir / "val").exists()
        assert (dataset_dir / "test").exists()

    def test_write_data_yaml(self, tmp_path: Path) -> None:
        """Test write_data_yaml creates valid data.yaml."""
        annotations_file = tmp_path / "annotations.json"
        images_dir = tmp_path / "images"
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        creator = DatasetCreator(
            annotations_file=annotations_file,
            images_dir=images_dir,
            dataset_dir=dataset_dir,
        )

        creator.classes = {0: "question", 1: "option"}

        with patch("digitex.ml.yolo.dataset.Path.cwd", return_value=tmp_path):
            creator.write_data_yaml()

        yaml_path = dataset_dir / "data.yaml"
        assert yaml_path.exists()

        with yaml_path.open() as f:
            data = yaml.safe_load(f)

        assert data["names"] == {0: "question", 1: "option"}
        assert data["train"] == "train"
        assert data["val"] == "val"
        assert data["test"] == "test"

    def test_create(self, tmp_path: Path) -> None:
        """Test create runs full dataset creation pipeline."""
        annotations_file = tmp_path / "annotations.json"
        images_dir = tmp_path / "images"
        dataset_dir = tmp_path / "dataset"

        annotations = [
            {
                "image": f"/data/local-files/?d=image{i}.jpg",
                "label": [
                    {
                        "polygonlabels": ["question"],
                        "points": [
                            [10.0, 20.0],
                            [50.0, 20.0],
                            [50.0, 80.0],
                            [10.0, 80.0],
                        ],
                    },
                ],
            }
            for i in range(5)
        ]
        annotations_file.write_text(json.dumps(annotations))
        images_dir.mkdir()

        for i in range(5):
            (images_dir / f"image{i}.jpg").write_bytes(b"fake")

        creator = DatasetCreator(
            annotations_file=annotations_file,
            images_dir=images_dir,
            dataset_dir=dataset_dir,
            train_split=0.6,
        )

        with patch("digitex.ml.yolo.dataset.Path.cwd", return_value=tmp_path):
            creator.create()

        assert (dataset_dir / "train").exists()
        assert (dataset_dir / "val").exists()
        assert (dataset_dir / "test").exists()
        assert (dataset_dir / "data.yaml").exists()
