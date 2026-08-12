"""Resolving conflicts when an extracted question collides with an existing file.

A `ConflictResolver` is just a callable that, given a `Conflict`, returns the
option number the new image actually belongs under. The default resolver keeps
the current option (no interaction).

The shape is a one-line type alias rather than a Protocol: there is one
resolver, and a callable is the smallest thing that expresses "given a
conflict, name the option". Add another resolver as a free function whenever a
second real adapter shows up. A caller reaches this seam by building a
configured `PageExtractor` and passing it to `BookExtractor`, which
`TestsExtractor` in turn accepts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image


@dataclass(frozen=True)
class Conflict:
    """An extracted question colliding with an already-saved file."""

    new_image: Image.Image
    existing_path: Path
    current_option: int


ConflictResolver = Callable[[Conflict], int]


def keep_current_option(conflict: Conflict) -> int:
    """Default resolver: trust the current option counter, no user interaction."""
    return conflict.current_option
