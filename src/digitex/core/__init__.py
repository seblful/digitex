"""Core functionality."""

from __future__ import annotations

import importlib
from typing import Any

from .schemas import Question, Session, Student, TestResult  # noqa: F401

_MODULES: dict[str, str] = {
    "TextExtractor": ".ocr",
}


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        module = importlib.import_module(_MODULES[name], __package__)
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = ["TextExtractor", "Question", "Session", "Student", "TestResult"]
