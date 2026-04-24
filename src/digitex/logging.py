"""Logging configuration using structlog."""

import logging
import sys
import codecs
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog

from digitex.config.settings import get_settings


def setup_logging(
    file_level: str | None = None,
    console_level: str | None = None,
    log_file: Path | None = None,
) -> None:
    """Configure structlog for the application."""
    cfg = get_settings()
    f_level = file_level or cfg.logging.file_level
    c_level = console_level or cfg.logging.console_level

    file_path = log_file or cfg.logging.log_file
    if not file_path.is_absolute():
        file_path = cfg.paths.root_dir / file_path

    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(file_path, maxBytes=10_485_760, backupCount=3, encoding="utf-8")
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
        if cfg.app.environment == "production"
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
