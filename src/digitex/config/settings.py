"""Application settings using Pydantic for configuration management."""

from functools import cached_property
from pathlib import Path
from typing import Self

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractionSettings(BaseSettings):
    """Image extraction settings."""

    model_config = SettingsConfigDict(env_prefix="EXTRACTION_")

    model_path: Path = Field(
        default=Path("extraction/models/page.pt"),
        description="Path to the YOLO segmentation model",
    )

    output_dir_name: str = Field(
        default="output",
        description="Subdirectory name for extracted images",
    )

    question_max_width: int = Field(
        default=2000,
        ge=1,
        description="Maximum width for extracted question images",
    )

    question_max_height: int = Field(
        default=2000,
        ge=1,
        description="Maximum height for extracted question images",
    )

    image_format: str = Field(
        default="jpg",
        description="Output image format (jpg, png, etc.)",
    )


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

    batch_size: int = Field(default=4, ge=1, description="Batch size for training")

    overlap_mask: bool = Field(
        default=False, description="Whether segmentation masks should overlap"
    )

    patience: int = Field(
        default=50, ge=1, description="Early stopping patience in epochs"
    )

    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")

    model_type: str = Field(default="seg", description="Type of YOLO model ('seg')")

    model_size: str = Field(
        default="m", description="Size of YOLO model ('n', 's', 'm', 'l', 'x')"
    )

    pretrained_model_path: str | None = Field(
        default=None, description="Path to a previously trained model"
    )

    model_subdir: str = Field(
        default="models", description="Subdirectory name for trained models"
    )

    runs_dir_name: str = Field(
        default="runs", description="Subdirectory name for training runs"
    )


class DataSettings(BaseSettings):
    """Data configuration for training."""

    model_config = SettingsConfigDict(env_prefix="DATA_")

    data_dir_name: str = Field(default="data", description="Subdirectory name for data")

    data_type_dir_name: str = Field(
        default="page",
        description="Type of task type (e.g., 'page', 'question', 'part')",
    )

    dataset_dir_name: str = Field(
        default="dataset", description="Subdirectory name for datasets"
    )

    images_dir_name: str = Field(
        default="images", description="Subdirectory name for images"
    )

    image_size: int = Field(
        default=640,
        ge=32,
        multiple_of=32,
        description="Input image size for training (must be multiple of 32)",
    )


class LabelStudioSettings(BaseSettings):
    """Label Studio connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="LABEL_STUDIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: str = Field(
        default="http://localhost:8080", description="Label Studio server URL"
    )

    api_key: str = Field(default="", description="Label Studio API key")

    image_size: int = Field(
        default=1280,
        ge=32,
        multiple_of=32,
        description="Image size for Label Studio tasks (must be multiple of 32)",
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
    def books_dir(self) -> Path:
        """Get the books directory path."""
        return self.home_dir / "books"

    @cached_property
    def extraction_dir(self) -> Path:
        """Get the extraction directory path."""
        return self.home_dir / "extraction"


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
    data: DataSettings = Field(default_factory=DataSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    label_studio: LabelStudioSettings = Field(default_factory=LabelStudioSettings)

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
