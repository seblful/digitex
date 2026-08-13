"""Settings every layer shares: environment, logging, timezone."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class AppSettings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore")

    environment: str = Field(
        default="development",
        # Compose sets the bare ``ENVIRONMENT``; without it in this list the
        # deployment sets that one variable and the field stays at its default,
        # so production never selects the JSON log renderer.
        validation_alias=AliasChoices("environment", "APP_ENVIRONMENT", "ENVIRONMENT"),
        description="Application environment (development, production)",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    model_config = SettingsConfigDict(env_prefix="LOGGING_", extra="ignore")

    file_level: LogLevel = Field(
        default="DEBUG",
        description="File logging level",
    )
    console_level: LogLevel = Field(
        default="INFO",
        description="Console logging level",
    )
    log_file: Path = Field(
        default=Path("logs/app.log"),
        description=(
            "Path to the log file. Relative paths resolve against the working"
            " directory — the container runs from /app with ./logs bind-mounted."
        ),
    )


class TimezoneSettings(BaseSettings):
    """Timezone configuration."""

    model_config = SettingsConfigDict(env_prefix="TIMEZONE_", extra="ignore")

    name: str = Field(
        default="Europe/Minsk",
        description="IANA timezone name (e.g. Europe/Minsk, Europe/Moscow)",
    )
