from __future__ import annotations

import re
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()


def natural_sort_key(path: Path) -> list[int | str]:
    parts: list[int | str] = []
    for chunk in re.split(r"(\d+)", path.stem):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk.lower())
    return parts


def get_tz() -> ZoneInfo:
    """Return the application timezone from settings."""
    from digitex.config import get_settings

    return ZoneInfo(get_settings().timezone.name)
