"""Tests for the YOLO Trainer class."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

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
