import logging
import os
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class Trainer:
    """YOLO model trainer for document segmentation tasks."""

    def __init__(
        self,
        dataset_dir: str | Path,
        model_type: str,
        model_size: str,
        num_epochs: int,
        image_size: int,
        batch_size: int,
        pretrained_model_path: str | Path | None = None,
        overlap_mask: bool = False,
        patience: int = 50,
        seed: int = 42,
    ) -> None:
        """Initialize the YOLO trainer.

        Args:
            dataset_dir: Directory containing the dataset and data.yaml.
            model_type: Type of YOLO model ('seg', 'obb', 'pose').
            model_size: Size of YOLO model ('n', 's', 'm', 'l', 'x').
            num_epochs: Number of training epochs.
            image_size: Input image size for training.
            batch_size: Batch size for training.
            pretrained_model_path: Path to a pre-trained model to fine-tune.
            overlap_mask: Whether to use overlapping masks for segmentation.
            patience: Early stopping patience in epochs.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If dataset_dir doesn't exist or model parameters are invalid.
        """
        self.dataset_dir = Path(dataset_dir)

        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")

        self.__data: str | None = None

        self.num_epochs = num_epochs
        self.image_size = image_size
        self.batch_size = batch_size
        self.overlap_mask = overlap_mask
        self.patience = patience
        self.seed = seed

        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info("CUDA device detected")
        else:
            self.device = 'cpu'
            logger.info("Using CPU device")

        self.device_count = torch.cuda.device_count()
        self.device_idxs = [i for i in range(self.device_count)]
        logger.info(f"Device count: {self.device_count}, devices: {self.device_idxs}")

        self.pretrained_model_path = str(pretrained_model_path) if pretrained_model_path else None
        self.model_yaml = f"yolo11{model_size}-{model_type}.yaml"
        self.model_pt = f"yolo11{model_size}-{model_type}.pt"

        self.__model: YOLO | None = None
        self.is_trained = False

    @property
    def model(self) -> YOLO:
        """Get or load the YOLO model.

        Returns:
            Loaded YOLO model.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        if self.__model is None:
            try:
                if self.pretrained_model_path is None:
                    model = YOLO(self.model_yaml).load(self.model_pt)
                    logger.info(f"Loaded pretrained model: {self.model_pt}")
                else:
                    model = YOLO(self.pretrained_model_path)
                    logger.info(f"Loaded custom pretrained model: {self.pretrained_model_path}")

                self.__model = model
            except Exception as e:
                raise RuntimeError(f"Failed to load YOLO model: {e}")

        return self.__model

    @property
    def data(self) -> str:
        """Get the data configuration path.

        Returns:
            Path to the data.yaml configuration file.

        Raises:
            FileNotFoundError: If data.yaml doesn't exist.
        """
        if self.__data is None:
            data_path = self.dataset_dir / "data.yaml"
            if not data_path.exists():
                raise FileNotFoundError(f"data.yaml not found in {self.dataset_dir}")
            self.__data = str(data_path)

        return self.__data

    def train(self) -> None:
        """Train the YOLO model.

        Raises:
            RuntimeError: If training fails.
        """
        try:
            logger.info("Starting training...")
            logger.info(f"Epochs: {self.num_epochs}, Image size: {self.image_size}, Batch size: {self.batch_size}")

            self.model.train(
                data=self.data,
                epochs=self.num_epochs,
                imgsz=self.image_size,
                batch=self.batch_size,
                overlap_mask=self.overlap_mask,
                patience=self.patience,
                device=self.device_idxs,
                seed=self.seed,
            )

            self.is_trained = True
            logger.info("Training completed successfully")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")

    def validate(self) -> None:
        """Validate the trained model on the test set.

        Raises:
            ValueError: If model has not been trained yet.
            RuntimeError: If validation fails.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before validating.")

        try:
            logger.info("Starting validation...")

            self.model.val(
                data=self.data,
                imgsz=self.image_size,
                split='test',
            )

            logger.info("Validation completed successfully")

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise RuntimeError(f"Validation failed: {e}")
