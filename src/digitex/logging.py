"""Logging configuration using structlog."""

import logging
from pathlib import Path

import structlog
from structlog.processors import Timestamp


def setup_logging(log_dir: Path | None = None, level: str = "INFO") -> None:
    """Configure structlog with project settings."""
    if log_dir is None:
        log_dir = Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.make_filtering_bound_logger(log_level),
            Timestamp(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )
