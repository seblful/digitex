"""Digitex - Document digitization toolkit."""

from __future__ import annotations

import importlib
from typing import Any

_MODULES: dict[str, str] = {
    "BookExtractor": ".extractors",
    "ExtractionResult": ".extractors",
    "LabelStudioClient": ".label_studio",
    "ManualExtractor": ".extractors",
    "PageDataCreator": ".creators",
    "PageExtractor": ".extractors",
    "TaskPredictor": ".label_studio",
}


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        module = importlib.import_module(_MODULES[name], __package__)
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = list(_MODULES.keys())
