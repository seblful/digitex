import logging
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.engine.model import Model

logger = logging.getLogger(__name__)


class Trainer:
    """YOLO model trainer for document segmentation tasks."""

    def __init__(
        self,
        train_config_path: str | Path,
        val_config_path: str | Path,
    ) -> None:
        """Initialize the YOLO trainer.

        Args:
            train_config_path: Path to training configuration YAML file.
            val_config_path: Path to validation configuration YAML file.

        Raises:
            ValueError: If config files don't exist.
        """
        self.train_config_path = Path(train_config_path)
        self.val_config_path = Path(val_config_path)

        if not self.train_config_path.exists():
            raise ValueError(f"Train config file not found: {self.train_config_path}")
        if not self.val_config_path.exists():
            raise ValueError(f"Val config file not found: {self.val_config_path}")

        self._model: Model | None = None
        self.is_trained = False

    def _load_config(self) -> dict:
        """Load training config from YAML file.

        Returns:
            Config dictionary.
        """
        with open(self.train_config_path) as f:
            return yaml.safe_load(f)

    @property
    def model(self) -> Model:
        """Get or load the YOLO model.

        Returns:
            Loaded YOLO model.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        if self._model is None:
            try:
                config = self._load_config()
                model_path = config.get("model")
                if not model_path:
                    raise ValueError("Config must contain 'model' key")

                self._model = YOLO(model_path)
                logger.info(f"Loaded model: {model_path}")

            except Exception as e:
                raise RuntimeError(f"Failed to load YOLO model: {e}")

        return self._model  # type: ignore[return-value]

    def train(self) -> None:
        """Train the YOLO model.

        Raises:
            RuntimeError: If training fails.
        """
        try:
            logger.info("Starting training...")

            self.model.train(
                cfg=self.train_config_path,
            )

            self.is_trained = True
            logger.info("Training completed successfully")

        except (RuntimeError, ValueError, FileNotFoundError) as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training failed: {e}") from e

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
                cfg=self.val_config_path,
            )

            logger.info("Validation completed successfully")

        except (RuntimeError, ValueError) as e:
            logger.error(f"Validation failed: {e}")
            raise RuntimeError(f"Validation failed: {e}") from e
