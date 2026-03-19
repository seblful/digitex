"""Configuration module for application settings."""

from .settings import get_settings, Settings, DatabaseSettings, TrainingSettings, PathsSettings

__all__ = [
    "get_settings",
    "Settings",
    "DatabaseSettings",
    "TrainingSettings",
    "PathsSettings",
]
