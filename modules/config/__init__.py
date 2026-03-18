"""Configuration module for application settings."""

from modules.config.settings import get_settings, Settings, AppSettings, DatabaseSettings, TrainingSettings, PathsSettings

__all__ = [
    "get_settings",
    "Settings",
    "AppSettings",
    "DatabaseSettings",
    "TrainingSettings",
    "PathsSettings",
]
