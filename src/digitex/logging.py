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


def setup_logging(settings: Settings) -> None:
    """Configure structlog for the application.

    Settings are injected rather than read here: each entry point resolves them
    once and passes the result down, so importing a module never reads the
    environment or installs a log handler as a side effect.
    """
    levels = logging.getLevelNamesMapping()
    f_level = levels[settings.logging.file_level]
    c_level = levels[settings.logging.console_level]

    # Relative to the working directory, which is what both callers want: a
    # laptop run from the checkout writes ./logs/, and the container's cwd is
    # /app with ./logs bind-mounted from the host. Neither needs to know where
    # the package itself was installed.
    file_path = settings.logging.log_file
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path

    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        file_path, maxBytes=10_485_760, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(f_level)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(c_level)
    console_handler.stream = codecs.getwriter("utf-8")(sys.stderr.buffer)

    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        handlers=[file_handler, console_handler],
    )

    renderer = (
        structlog.processors.JSONRenderer()
        if settings.app.environment == "production"
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            renderer,
        ],
        # structlog turns calls below this into no-ops before any handler runs,
        # so it has to pass whatever the more verbose destination wants; each
        # handler then filters down to its own level.
        wrapper_class=structlog.make_filtering_bound_logger(min(f_level, c_level)),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
