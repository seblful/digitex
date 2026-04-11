import logging
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.model import Model

logger = logging.getLogger(__name__)


class Trainer:
    """YOLO model trainer for document segmentation tasks."""

    def __init__(
        self,
        config_path: str | Path = "training/config.yaml",
    ) -> None:
        """Initialize the YOLO trainer.

        Args:
            config_path: Path to training configuration YAML file.

        Raises:
            ValueError: If config_path doesn't exist.
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise ValueError(f"Config file not found: {self.config_path}")

        self._model: Model | None = None
        self.is_trained = False

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
                model = YOLO(self.config_path)
                logger.info(f"Loaded model from config: {self.config_path}")

                self._model = model
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
                cfg=self.config_path,
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
                split="test",
            )

            logger.info("Validation completed successfully")

        except (RuntimeError, ValueError) as e:
            logger.error(f"Validation failed: {e}")
            raise RuntimeError(f"Validation failed: {e}") from e
