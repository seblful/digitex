"""Tests for ``setup_logging``.

The log file is the assertion surface: structlog's own level filter runs before
any handler, so "is this event in ``app.log``" is the only question that proves
the two level knobs are wired to the two destinations.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from digitex.config.settings import (
    LoggingSettings,
    LogLevel,
    PathsSettings,
    Settings,
)
from digitex.logging import setup_logging


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Hand back the global logging and structlog state each test borrows."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)
    structlog.reset_defaults()


def _configure(settings: Settings) -> None:
    """Run ``setup_logging`` against a bare root logger, as a CLI entry point does.

    ``setup_logging`` configures via ``logging.basicConfig``, which no-ops once
    the root logger has handlers — and pytest's logging plugin installs one
    around every test phase. Clearing here reproduces the production start.
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    setup_logging(settings)


def _settings(
    tmp_path: Path, *, file_level: LogLevel, console_level: LogLevel
) -> Settings:
    return Settings(
        paths=PathsSettings(root_dir=tmp_path),
        logging=LoggingSettings(
            file_level=file_level,
            console_level=console_level,
            log_file=Path("logs/app.log"),
        ),
    )


def _emit_and_read(settings: Settings, tmp_path: Path) -> str:
    _configure(settings)
    logger = structlog.get_logger()
    logger.debug("debug_event")
    logger.info("info_event")
    for handler in logging.getLogger().handlers:
        handler.flush()
    return (tmp_path / "logs" / "app.log").read_text(encoding="utf-8")


class TestFileLevelIsHonored:
    def test_debug_reaches_the_file_when_file_level_is_debug(
        self, tmp_path: Path
    ) -> None:
        contents = _emit_and_read(
            _settings(tmp_path, file_level="DEBUG", console_level="INFO"), tmp_path
        )

        assert "debug_event" in contents

    def test_info_reaches_the_file_too(self, tmp_path: Path) -> None:
        contents = _emit_and_read(
            _settings(tmp_path, file_level="DEBUG", console_level="INFO"), tmp_path
        )

        assert "info_event" in contents

    def test_debug_stays_out_of_the_file_when_file_level_is_info(
        self, tmp_path: Path
    ) -> None:
        contents = _emit_and_read(
            _settings(tmp_path, file_level="INFO", console_level="INFO"), tmp_path
        )

        assert "debug_event" not in contents
        assert "info_event" in contents


class TestLogFileLocation:
    def test_relative_log_file_resolves_under_root_dir(self, tmp_path: Path) -> None:
        _configure(_settings(tmp_path, file_level="INFO", console_level="INFO"))

        assert (tmp_path / "logs" / "app.log").exists()
