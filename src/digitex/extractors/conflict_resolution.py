"""Conflict resolution strategies for PageExtractor."""

from pathlib import Path
from typing import Protocol

from PIL import Image


class ConflictResolutionStrategy(Protocol):
    """Protocol for conflict resolution strategies."""

    def resolve(
        self,
        new_image: Image.Image,
        existing_path: Path,
        current_option: int,
        source_image_name: str,
    ) -> int:
        """Resolve a file conflict.

        Args:
            new_image: The new image that would be saved.
            existing_path: Path of the existing file.
            current_option: Current option number.
            source_image_name: Name of the source image file.

        Returns:
            The correct option number.
        """
        ...


class AutoConflictResolution:
    """Automatic conflict resolution that keeps the current option."""

    def resolve(
        self,
        new_image: Image.Image,
        existing_path: Path,
        current_option: int,
        source_image_name: str,
    ) -> int:
        """Return current option without user interaction."""
        return current_option


class InteractiveConflictResolution:
    """Interactive conflict resolution that prompts the user."""

    def resolve(
        self,
        new_image: Image.Image,
        existing_path: Path,
        current_option: int,
        source_image_name: str,
    ) -> int:
        """Show image and prompt user for correct option."""
        new_image.show()
        print(f"Image: {source_image_name}")
        print(f"Current option: {current_option}")

        while True:
            user_input = input(
                f"Enter correct option number (current: {current_option}): "
            ).strip()
            if user_input.isdigit():
                option = int(user_input)
                if 1 <= option <= 10:
                    return option
            print("Please enter a number between 1 and 10")
