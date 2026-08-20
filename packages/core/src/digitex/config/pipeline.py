"""Settings for the local-only workflows: extraction, training, annotation.

Grouped behind :class:`PipelineSettings` rather than sitting flat on
:class:`~digitex.config.settings.Settings`, so that reading
``settings.pipeline.openrouter`` says which layer the value belongs to. Nothing
the deployed bot runs touches this group — the packages it configures all live in
`digitex-studio`, which the production image does not install.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — pydantic resolves the annotation at runtime

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractionSettings(BaseSettings):
    """Image extraction settings."""

    model_config = SettingsConfigDict(env_prefix="EXTRACTION_", extra="ignore")

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


class OpenRouterSettings(BaseSettings):
    """OpenRouter API settings."""

    model_config = SettingsConfigDict(env_prefix="OPENROUTER_", extra="ignore")

    api_key: str = Field(
        default="",
        description="OpenRouter API key",
    )

    model: str = Field(
        default="google/gemini-3-flash-preview",
        description="Model for answer extraction via OpenRouter",
    )

    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )


class DataSettings(BaseSettings):
    """Data configuration for training."""

    model_config = SettingsConfigDict(env_prefix="DATA_", extra="ignore")

    dataset_dir_name: str = Field(
        default="dataset", description="Subdirectory name for datasets"
    )

    images_dir_name: str = Field(
        default="images", description="Subdirectory name for images"
    )

    image_size: int = Field(
        default=1280,
        ge=32,
        multiple_of=32,
        description="Input image size for training (must be multiple of 32)",
    )


class LabelStudioSettings(BaseSettings):
    """Label Studio connection settings."""

    model_config = SettingsConfigDict(env_prefix="LABEL_STUDIO_", extra="ignore")

    url: str = Field(
        default="http://localhost:8080", description="Label Studio server URL"
    )

    api_key: str = Field(default="", description="Label Studio API key")

    # The server reads this one itself — it is what a ``?d=`` URI in a task is
    # relative to — so a repair that has to tell a reachable image from a
    # stranded one asks for the same value rather than a second opinion.
    local_files_document_root: Path | None = Field(
        default=None,
        description="Directory the server serves local files from",
    )


class PipelineSettings(BaseSettings):
    """Every setting only the local workflows read."""

    model_config = SettingsConfigDict(extra="ignore")

    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    label_studio: LabelStudioSettings = Field(default_factory=LabelStudioSettings)
    data: DataSettings = Field(default_factory=DataSettings)
