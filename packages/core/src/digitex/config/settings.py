"""Composition and the one process-wide cache.

Entry points call :func:`get_settings` once and thread the result down, so
importing a module never reads the environment or a file.
"""

from __future__ import annotations

from threading import Lock
from typing import Self

from dotenv import find_dotenv, load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from digitex.config.bot import BotSettings
from digitex.config.database import DatabaseSettings
from digitex.config.paths import PathsSettings
from digitex.config.pipeline import PipelineSettings
from digitex.config.runtime import AppSettings, LoggingSettings, TimezoneSettings


def _load_env() -> None:
    """Load this machine's ``.env``, if it has one.

    One file per machine — the laptop's carries development values, the
    server's production ones — so nothing has to be kept in sync with a
    second copy. Real environment variables win over the file: Compose passes
    ``DATABASE_URL`` and ``ENVIRONMENT`` that way, and CI passes everything
    that way.

    Searched for upwards from the working directory rather than beside the
    package, which is what makes an installed wheel workable: the container has
    no ``.env`` at all and gets every value from Compose, while a checkout is
    found from any subdirectory of it.
    """
    env_file = find_dotenv(usecwd=True)
    if env_file:
        load_dotenv(env_file, override=False)


class Settings(BaseSettings):
    """Every setting, grouped by the layer that reads it.

    ``pipeline`` holds the groups only the local workflows touch. It is a plain
    field rather than something lazy so tests can inject one, and because every
    field inside it has a default — the deployed bot constructing the group
    costs an environment read, not a chance of failing to start.
    """

    model_config = SettingsConfigDict(extra="ignore")

    app: AppSettings = Field(default_factory=AppSettings)
    bot: BotSettings = Field(default_factory=BotSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    timezone: TimezoneSettings = Field(default_factory=TimezoneSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)

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


def reset_settings_cache() -> None:
    """Drop the cached Settings so the next :func:`get_settings` re-reads.

    For tests that change the environment — pointing the suite at a throwaway
    Postgres, say — after something has already resolved settings. Production
    code has no reason to call this.
    """
    global _settings  # noqa: PLW0603 — same cache as get_settings

    with _settings_lock:
        _settings = None
