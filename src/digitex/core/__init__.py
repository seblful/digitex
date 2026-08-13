"""Core functionality.

``TextExtractor`` is imported lazily so that ``import digitex.core`` does not
pull in pytesseract. The ``TYPE_CHECKING`` import keeps it a real type for
callers that annotate with it — ``__getattr__`` alone resolves to ``Any``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .ocr import TextExtractor

_MODULES: dict[str, str] = {
    "TextExtractor": ".ocr",
}


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        module = importlib.import_module(_MODULES[name], __package__)
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = ["TextExtractor"]
