"""Configuration module for application settings."""

from .settings import (
    DatabaseSettings,
    ExtractionSettings,
    OCRSettings,
    PDFSettings,
    PathsSettings,
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
    "PDFSettings",
    "PathsSettings",
    "TrainingSettings",
]
