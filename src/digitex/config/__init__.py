"""Configuration module for application settings."""

from .settings import (
    DatabaseSettings,
    ExtractionSettings,
    OCRSettings,
    PathsSettings,
    ProcessingSettings,
    Settings,
    TrainingSettings,
    get_settings,
)

__all__ = [
    "get_settings",
    "Settings",
    "DatabaseSettings",
    "ExtractionSettings",
    "OCRSettings",
    "PathsSettings",
    "ProcessingSettings",
    "TrainingSettings",
]
