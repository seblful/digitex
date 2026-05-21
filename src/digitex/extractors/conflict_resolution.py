"""Resolving conflicts when an extracted question collides with an existing file.

A `ConflictResolver` is just a callable that, given a `Conflict`, returns the
option number the new image actually belongs under. The default resolver keeps
the current option (no interaction); `prompt_user` shows the image and asks.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

Prompter = Callable[[str], str]


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


def prompt_user(conflict: Conflict, *, prompter: Prompter = input) -> int:
    """Interactive resolver: show the image and ask which option it belongs to."""
    conflict.new_image.show()
    print(f"Image: {conflict.source_image_name}")
    print(f"Current option: {conflict.current_option}")
    while True:
        user_input = prompter(
            f"Enter correct option number (current: {conflict.current_option}): "
        ).strip()
        if user_input.isdigit():
            option = int(user_input)
            if 1 <= option <= 10:
                return option
        print("Please enter a number between 1 and 10")
