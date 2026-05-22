"""Resolving conflicts when an extracted question collides with an existing file.

A `ConflictResolver` is just a callable that, given a `Conflict`, returns the
option number the new image actually belongs under. The default resolver keeps
the current option (no interaction).

Per ADR 0002, the shape is a one-line type alias rather than a Protocol. Add
another resolver as a free function whenever a second real adapter shows up.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image


@dataclass
class Conflict:
    """An extracted question colliding with an already-saved file."""

    new_image: Image.Image
    existing_path: Path
    current_option: int
    source_image_name: str


ConflictResolver = Callable[[Conflict], int]


def keep_current_option(conflict: Conflict) -> int:
    """Default resolver: trust the current option counter, no user interaction."""
    return conflict.current_option
