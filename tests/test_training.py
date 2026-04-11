"""Tests for the ML YOLO training module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from digitex.ml.yolo.dataset import DatasetCreator
from digitex.ml.yolo.trainer import Trainer


class TestTrainer:
    """Test suite for Trainer class."""

    def test_init_validates_train_config_exists(self, tmp_path: Path) -> None:
        """Test initialization raises ValueError when train config doesn't exist."""
        train_config = tmp_path / "train.yaml"
        val_config = tmp_path / "val.yaml"
        val_config.touch()

        with pytest.raises(ValueError, match="Train config file not found"):
            Trainer(
                train_config_path=str(train_config), val_config_path=str(val_config)
            )

    def test_init_validates_val_config_exists(self, tmp_path: Path) -> None:
        """Test initialization raises ValueError when val config doesn't exist."""
        train_config = tmp_path / "train.yaml"
        val_config = tmp_path / "val.yaml"
        train_config.touch()

        with pytest.raises(ValueError, match="Val config file not found"):
            Trainer(
                train_config_path=str(train_config), val_config_path=str(val_config)
            )

    def test_init_success(self, tmp_path: Path) -> None:
        """Test successful initialization."""
        train_config = tmp_path / "train.yaml"
        val_config = tmp_path / "val.yaml"
        train_config.write_text("model: yolov8n.pt\n")
        val_config.write_text("model: yolov8n.pt\n")

        trainer = Trainer(
            train_config_path=str(train_config), val_config_path=str(val_config)
        )

        assert trainer.train_config_path == train_config
        assert trainer.val_config_path == val_config
        assert trainer._model is None
        assert trainer.is_trained is False

    def test_load_config(self, tmp_path: Path) -> None:
        """Test _load_config parses YAML correctly."""
        train_config = tmp_path / "train.yaml"
        val_config = tmp_path / "val.yaml"
        config_data = {"model": "yolov8n.pt", "epochs": 100, "imgsz": 640}
        train_config.write_text(yaml.dump(config_data))
        val_config.write_text("model: yolov8n.pt\n")

        trainer = Trainer(
            train_config_path=str(train_config), val_config_path=str(val_config)
        )

        loaded = trainer._load_config()
        assert loaded["model"] == "yolov8n.pt"
        assert loaded["epochs"] == 100
        assert loaded["imgsz"] == 640

    def test_model_loads_yolo(self, tmp_path: Path) -> None:
        """Test model property loads YOLO model."""
        train_config = tmp_path / "train.yaml"
        val_config = tmp_path / "val.yaml"
        config_data = {"model": "yolov8n.pt"}
        train_config.write_text(yaml.dump(config_data))
        val_config.write_text("model: yolov8n.pt\n")

        trainer = Trainer(
            train_config_path=str(train_config), val_config_path=str(val_config)
        )

        with patch("digitex.ml.yolo.trainer.YOLO") as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model

            model = trainer.model

            mock_yolo.assert_called_once_with("yolov8n.pt")
            assert model is mock_model

    def test_model_caches_after_first_access(self, tmp_path: Path) -> None:
        """Test model is cached after first access."""
        train_config = tmp_path / "train.yaml"
        val_config = tmp_path / "val.yaml"
        config_data = {"model": "yolov8n.pt"}
        train_config.write_text(yaml.dump(config_data))
        val_config.write_text("model: yolov8n.pt\n")

        trainer = Trainer(
            train_config_path=str(train_config), val_config_path=str(val_config)
        )

        with patch("digitex.ml.yolo.trainer.YOLO") as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model

            model1 = trainer.model
            model2 = trainer.model

            assert model1 is model2
            mock_yolo.assert_called_once()

    def test_model_raises_on_load_failure(self, tmp_path: Path) -> None:
        """Test model property raises RuntimeError when YOLO fails to load."""
        train_config = tmp_path / "train.yaml"
        val_config = tmp_path / "val.yaml"
        config_data = {"model": "invalid.pt"}
        train_config.write_text(yaml.dump(config_data))
        val_config.write_text("model: invalid.pt\n")

        trainer = Trainer(
            train_config_path=str(train_config), val_config_path=str(val_config)
        )

        with patch("digitex.ml.yolo.trainer.YOLO") as mock_yolo:
            mock_yolo.side_effect = FileNotFoundError("Model not found")

            with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
                _ = trainer.model

    def test_train_success(self, tmp_path: Path) -> None:
        """Test successful training."""
        train_config = tmp_path / "train.yaml"
        val_config = tmp_path / "val.yaml"
        config_data = {"model": "yolov8n.pt"}
        train_config.write_text(yaml.dump(config_data))
        val_config.write_text("model: yolov8n.pt\n")

        trainer = Trainer(
            train_config_path=str(train_config), val_config_path=str(val_config)
        )

        with patch("digitex.ml.yolo.trainer.YOLO") as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model

            trainer.train()

            assert trainer.is_trained is True
            mock_model.train.assert_called_once()

    def test_train_raises_on_error(self, tmp_path: Path) -> None:
        """Test train raises RuntimeError on training failure."""
        train_config = tmp_path / "train.yaml"
        val_config = tmp_path / "val.yaml"
        config_data = {"model": "yolov8n.pt"}
        train_config.write_text(yaml.dump(config_data))
        val_config.write_text("model: yolov8n.pt\n")

        trainer = Trainer(
            train_config_path=str(train_config), val_config_path=str(val_config)
        )

        with patch("digitex.ml.yolo.trainer.YOLO") as mock_yolo:
            mock_model = MagicMock()
            mock_model.train.side_effect = RuntimeError("Training failed")
            mock_yolo.return_value = mock_model

            with pytest.raises(RuntimeError, match="Training failed"):
                trainer.train()

    def test_validate_raises_when_not_trained(self, tmp_path: Path) -> None:
        """Test validate raises ValueError when model hasn't been trained."""
        train_config = tmp_path / "train.yaml"
        val_config = tmp_path / "val.yaml"
        config_data = {"model": "yolov8n.pt"}
        train_config.write_text(yaml.dump(config_data))
        val_config.write_text("model: yolov8n.pt\n")

        trainer = Trainer(
            train_config_path=str(train_config), val_config_path=str(val_config)
        )

        with pytest.raises(ValueError, match="Model must be trained before validating"):
            trainer.validate()

    def test_validate_success(self, tmp_path: Path) -> None:
        """Test successful validation after training."""
        train_config = tmp_path / "train.yaml"
        val_config = tmp_path / "val.yaml"
        config_data = {"model": "yolov8n.pt"}
        train_config.write_text(yaml.dump(config_data))
        val_config.write_text("model: yolov8n.pt\n")

        trainer = Trainer(
            train_config_path=str(train_config), val_config_path=str(val_config)
        )

        with patch("digitex.ml.yolo.trainer.YOLO") as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model

            trainer.train()
            trainer.validate()

            mock_model.val.assert_called_once()


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
