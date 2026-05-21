"""Application settings using Pydantic for configuration management."""

import os
from functools import cached_property
from pathlib import Path
from threading import Lock
from typing import Literal, Self

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


def _load_env() -> None:
    env_name = os.environ.get("ENVIRONMENT") or os.environ.get(
        "APP_ENVIRONMENT", "development"
    )
    env_specific = BASE_DIR / f".env.{env_name}"
    if env_specific.exists():
        load_dotenv(env_specific, override=False)

    env_file = BASE_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)


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


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection settings.

    The DSN is read from the ``DATABASE_URL`` env var (12-factor convention)
    or ``DB_DSN`` as a fallback.
    """

    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")

    dsn: PostgresDsn = Field(
        default=PostgresDsn("postgresql://digitex:digitex@localhost:5432/digitex"),
        validation_alias=AliasChoices("DB_DSN", "DATABASE_URL"),
        description="PostgreSQL connection string.",
    )
    pool_min_size: int = Field(default=2, ge=1)
    pool_max_size: int = Field(default=10, ge=1)
    pool_timeout: float = Field(default=10.0, gt=0)
    statement_timeout_ms: int = Field(
        default=5000,
        ge=0,
        description="Server-side statement timeout in milliseconds.",
    )
    idle_in_transaction_timeout_ms: int = Field(
        default=10000,
        ge=0,
        description="Server-side idle-in-transaction timeout in milliseconds.",
    )
    sslmode: (
        Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
        | None
    ) = Field(
        default=None,
        description="If set, appended to the DSN as ?sslmode=...",
    )

    @computed_field
    @property
    def conninfo(self) -> str:
        """DSN as a libpq conninfo string suitable for psycopg/AsyncConnectionPool."""
        dsn_str = str(self.dsn)
        if self.sslmode is None:
            return dsn_str
        sep = "&" if "?" in dsn_str else "?"
        return f"{dsn_str}{sep}sslmode={self.sslmode}"

    @computed_field
    @property
    def server_options(self) -> str:
        """Libpq ``options`` parameter setting statement + idle-in-tx timeouts."""
        idle_ms = self.idle_in_transaction_timeout_ms
        return (
            f"-c statement_timeout={self.statement_timeout_ms}"
            f" -c idle_in_transaction_session_timeout={idle_ms}"
        )


class TrainingSettings(BaseSettings):
    """YOLO model training parameters."""

    model_config = SettingsConfigDict(env_prefix="TRAIN_", extra="ignore")

    runs_dir_name: str = Field(
        default="runs", description="Subdirectory name for training runs"
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


class BotSettings(BaseSettings):
    """Telegram bot settings."""

    model_config = SettingsConfigDict(env_prefix="BOT_", extra="ignore")

    token: str = Field(
        default="",
        description="Telegram bot token from @BotFather",
    )

    admin_user_id: int = Field(
        default=0,
        description="Telegram user ID of the bot admin who approves registrations",
    )


class AppSettings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore")

    environment: str = Field(
        default="development",
        description="Application environment (development, production)",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    model_config = SettingsConfigDict(env_prefix="LOGGING_", extra="ignore")

    file_level: str = Field(
        default="DEBUG",
        description="File logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    console_level: str = Field(
        default="INFO",
        description="Console logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    log_file: Path = Field(
        default=Path("logs/app.log"),
        description="Path to the log file",
    )


class TimezoneSettings(BaseSettings):
    """Timezone configuration."""

    model_config = SettingsConfigDict(env_prefix="TIMEZONE_", extra="ignore")

    name: str = Field(
        default="Europe/Minsk",
        description="IANA timezone name (e.g. Europe/Minsk, Europe/Moscow)",
    )


class PathsSettings(BaseSettings):
    """Directory path settings.

    All paths derive from root_dir, which defaults to cwd() and can be
    overridden via the PATH_ROOT_DIR environment variable.
    """

    model_config = SettingsConfigDict(env_prefix="PATH_", extra="ignore")

    root_dir: Path = Field(default_factory=Path.cwd)

    # Top-level directories

    @computed_field
    @cached_property
    def training_dir(self) -> Path:
        return self.root_dir / "training"

    @computed_field
    @cached_property
    def books_dir(self) -> Path:
        return self.root_dir / "books"

    @computed_field
    @cached_property
    def extraction_dir(self) -> Path:
        return self.root_dir / "extraction"

    # Extraction sub-paths

    @computed_field
    @cached_property
    def extraction_data_dir(self) -> Path:
        return self.extraction_dir / "data"

    @computed_field
    @cached_property
    def extraction_output_dir(self) -> Path:
        return self.extraction_data_dir / "output"

    @computed_field
    @cached_property
    def extraction_manual_dir(self) -> Path:
        return self.extraction_data_dir / "manual"

    @computed_field
    @cached_property
    def extraction_model_path(self) -> Path:
        return self.extraction_dir / "models" / "page.pt"

    # Training sub-paths

    @computed_field
    @cached_property
    def training_data_dir(self) -> Path:
        return self.training_dir / "data"

    @computed_field
    @cached_property
    def training_configs_dir(self) -> Path:
        return self.training_dir / "configs"


class Settings(BaseSettings):
    """Main settings class that composes all settings categories."""

    model_config = SettingsConfigDict(extra="ignore")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    label_studio: LabelStudioSettings = Field(default_factory=LabelStudioSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    app: AppSettings = Field(default_factory=AppSettings)
    bot: BotSettings = Field(default_factory=BotSettings)
    timezone: TimezoneSettings = Field(default_factory=TimezoneSettings)

    @classmethod
    def load(cls) -> Self:
        _load_env()
        return cls()


_settings: Settings | None = None
_settings_lock = Lock()


def get_settings() -> Settings:
    global _settings  # noqa: PLW0603 — module-level cache is the intended pattern

    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = Settings.load()

    return _settings
