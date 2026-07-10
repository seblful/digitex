"""Label Studio integration."""

from __future__ import annotations

import importlib
from typing import Any

# Lazy imports keep ``label_studio.geometry`` importable without pulling in
# the Label Studio SDK and the YOLO predictor stack.
_MODULES: dict[str, str] = {
    "LabelStudioClient": ".client",
    "TaskPredictor": ".predictor",
}


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        module = importlib.import_module(_MODULES[name], __package__)
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = list(_MODULES.keys())
