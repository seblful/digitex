"""Application settings using Pydantic for configuration management."""

from functools import cached_property
from pathlib import Path
from threading import Lock
from typing import Literal, Self

from dotenv import load_dotenv
from pydantic import (
    AliasChoices,
    Field,
    PostgresDsn,
    computed_field,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _load_env() -> None:
    """Load this machine's ``.env``, if it has one.

    One file per machine — the laptop's carries development values, the
    server's production ones — so nothing has to be kept in sync with a
    second copy. Real environment variables win over the file: Compose passes
    ``DATABASE_URL`` and ``ENVIRONMENT`` that way, and CI passes everything
    that way.
    """
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
        validation_alias=AliasChoices("dsn", "DB_DSN", "DATABASE_URL"),
        description="PostgreSQL connection string.",
    )
    pool_min_size: int = Field(default=2, ge=1)
    pool_max_size: int = Field(default=10, ge=1)
    pool_timeout: float = Field(default=10.0, gt=0)
    connect_timeout: int = Field(
        default=10, gt=0, description="TCP connect timeout in seconds."
    )
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
        params: list[str] = [f"connect_timeout={self.connect_timeout}"]
        if self.sslmode is not None:
            params.append(f"sslmode={self.sslmode}")
        sep = "&" if "?" in dsn_str else "?"
        return f"{dsn_str}{sep}{'&'.join(params)}"

    @computed_field
    @property
    def server_options(self) -> str:
        """Libpq ``options`` parameter setting statement + idle-in-tx timeouts."""
        idle_ms = self.idle_in_transaction_timeout_ms
        return (
            f"-c statement_timeout={self.statement_timeout_ms}"
            f" -c idle_in_transaction_session_timeout={idle_ms}"
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
    """Where the non-code inputs and outputs live.

    Everything the project reads or writes that is not source — the book
    archive, the extraction output tree, model weights, training data — sits
    under ``data_root``. Code never derives a data path from its own location,
    so the package behaves the same installed into a container as it does run
    out of a checkout.

    ``data_root`` is resolved against the working directory, which is
    deliberate: an installed package cannot find the checkout it was built
    from, and guessing is exactly what the old ``BASE_DIR`` did. Set
    ``PATH_DATA_ROOT`` to be explicit. Every command that needs a directory
    reports the resolved path when it is missing, so the wrong cwd fails loudly
    instead of quietly extracting nothing.
    """

    model_config = SettingsConfigDict(env_prefix="PATH_", extra="ignore")

    data_root: Path = Field(
        default=Path("var"),
        description="Root of every non-code input and output.",
    )

    # Where the bot resolves an image's object_key against. Production rsyncs
    # the corpus to a directory of its own and bind-mounts it, so this is not
    # derivable from data_root there; unset, it is the extraction output tree
    # the keys were written from, which is what a laptop wants.
    questions_dir: Path | None = None

    # Repo content, not data: these YAMLs are hand-tuned hyperparameters under
    # version control, so they stay in the checkout rather than the data root.
    training_configs_dir: Path = Field(
        default=Path("configs/training"),
        description="Holds the {name}_train.yaml / {name}_val.yaml pair.",
    )

    @field_validator("data_root", "questions_dir", "training_configs_dir")
    @classmethod
    def _absolute(cls, value: Path | None) -> Path | None:
        """Pin relative paths to the working directory once, at load time.

        Resolving here rather than at each use means an error names a real
        absolute path, and nothing later depends on the cwd staying put.
        """
        return value.resolve() if value is not None else None

    # Top-level directories

    @computed_field
    @cached_property
    def books_dir(self) -> Path:
        return self.data_root / "books"

    @computed_field
    @cached_property
    def models_dir(self) -> Path:
        return self.data_root / "models"

    @computed_field
    @cached_property
    def extraction_dir(self) -> Path:
        return self.data_root / "extraction"

    # Extraction sub-paths

    @computed_field
    @cached_property
    def extraction_output_dir(self) -> Path:
        return self.extraction_dir / "output"

    @computed_field
    @cached_property
    def question_images_dir(self) -> Path:
        """Root that a question image's stored ``object_key`` resolves against."""
        return self.questions_dir or self.extraction_output_dir

    @computed_field
    @cached_property
    def extraction_model_path(self) -> Path:
        return self.models_dir / "page.pt"

    # Training sub-paths

    @computed_field
    @cached_property
    def training_data_dir(self) -> Path:
        return self.data_root / "training" / "data"


class Settings(BaseSettings):
    """Main settings class that composes all settings categories."""

    model_config = SettingsConfigDict(extra="ignore")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
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
