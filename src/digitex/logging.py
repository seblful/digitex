"""Logging configuration using structlog."""

from __future__ import annotations

import codecs
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from digitex.config.settings import Settings


def setup_logging(settings: Settings) -> None:
    """Configure structlog for the application.

    Per ADR 0001, settings are injected at the outermost module boundary. The
    CLI entry points call ``get_settings()`` once and pass the result here.
    """
    f_level = settings.logging.file_level
    c_level = settings.logging.console_level

    file_path = settings.logging.log_file
    if not file_path.is_absolute():
        file_path = settings.paths.root_dir / file_path

    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        file_path, maxBytes=10_485_760, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, f_level))

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, c_level))
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
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, c_level)),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
