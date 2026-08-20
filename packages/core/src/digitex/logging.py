"""Logging configuration using structlog."""

from __future__ import annotations

import codecs
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from digitex.config import Settings

_LOG_FILE_MAX_BYTES = 10_485_760  # 10 MiB
_LOG_FILE_BACKUPS = 3


def setup_logging(settings: Settings) -> None:
    """Configure structlog for the application.

    Settings are injected rather than read here: each entry point resolves them
    once and passes the result down, so importing a module never reads the
    environment or installs a log handler as a side effect.
    """
    levels = logging.getLevelNamesMapping()
    file_level = levels[settings.logging.file_level]
    console_level = levels[settings.logging.console_level]

    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        handlers=[
            _file_handler(settings.logging.log_file, file_level),
            _console_handler(console_level),
        ],
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            _renderer(production=settings.app.environment == "production"),
        ],
        # structlog turns calls below this into no-ops before any handler runs, so
        # it has to pass whatever the more verbose destination wants; each handler
        # then filters down to its own level.
        wrapper_class=structlog.make_filtering_bound_logger(
            min(file_level, console_level)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _file_handler(log_file: Path, level: int) -> logging.Handler:
    """A rotating handler, with its directory created.

    The path is relative to the working directory, which is what both callers
    want: a laptop run from the checkout writes ./logs/, and the container's cwd
    is /app with ./logs bind-mounted from the host. Neither needs to know where
    the package itself was installed.
    """
    path = log_file if log_file.is_absolute() else Path.cwd() / log_file
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        path,
        maxBytes=_LOG_FILE_MAX_BYTES,
        backupCount=_LOG_FILE_BACKUPS,
        encoding="utf-8",
    )
    handler.setLevel(level)
    return handler


def _console_handler(level: int) -> logging.Handler:
    """stderr, wrapped so a Cyrillic log line survives a cp1251 console."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.stream = codecs.getwriter("utf-8")(sys.stderr.buffer)
    return handler


def _renderer(*, production: bool) -> structlog.typing.Processor:
    """JSON where something collects logs, colour where a person reads them."""
    if production:
        return structlog.processors.JSONRenderer()
    return structlog.dev.ConsoleRenderer(colors=True)
