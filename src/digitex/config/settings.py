"""Application settings using Pydantic for configuration management."""

from functools import cached_property
from pathlib import Path
from typing import Self

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    path: str = Field(
        default="data/tests.db", description="Path to the SQLite database file"
    )


class TrainingSettings(BaseSettings):
    """YOLO model training parameters."""

    model_config = SettingsConfigDict(env_prefix="TRAIN_")

    num_epochs: int = Field(default=100, ge=1, description="Number of training epochs")

    image_size: int = Field(
        default=640,
        ge=32,
        multiple_of=32,
        description="Input image size for training (must be multiple of 32)",
    )

    batch_size: int = Field(default=4, ge=1, description="Batch size for training")

    overlap_mask: bool = Field(
        default=False, description="Whether segmentation masks should overlap"
    )

    patience: int = Field(
        default=50, ge=1, description="Early stopping patience in epochs"
    )

    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")

    data_subdir: str = Field(
        default="page",
        description="Type of task type (e.g., 'page', 'question', 'part')",
    )

    model_type: str = Field(default="seg", description="Type of YOLO model ('seg')")

    model_size: str = Field(
        default="m", description="Size of YOLO model ('n', 's', 'm', 'l', 'x')"
    )

    pretrained_model_path: str | None = Field(
        default=None, description="Path to a previously trained model"
    )


class PathsSettings(BaseSettings):
    """Directory path settings."""

    model_config = SettingsConfigDict(env_prefix="PATH_")

    @cached_property
    def home_dir(self) -> Path:
        """Get the current working directory."""
        return Path.cwd()

    @cached_property
    def training_dir(self) -> Path:
        """Get the training directory path."""
        return self.home_dir / "training"

    @cached_property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        return self.training_dir / "data"

    @cached_property
    def dataset_dir(self) -> Path:
        """Get the dataset directory path."""
        return self.data_dir / "dataset"

    @cached_property
    def model_dir(self) -> Path:
        """Get the model directory path."""
        return self.training_dir / "models"

    @cached_property
    def raw_data_dir(self) -> Path:
        """Get the raw data directory path."""
        return self.home_dir / "books"


class Settings(BaseSettings):
    """Main settings class that composes all settings categories."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)

    @classmethod
    def load(cls) -> Self:
        """Load settings from environment and default values."""
        return cls()


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the singleton settings instance.

    Returns:
        Settings: The global settings instance.
    """
    global _settings

    if _settings is None:
        _settings = Settings.load()

    return _settings
