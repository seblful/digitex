import os
from urllib.parse import unquote

from tqdm import tqdm

from digitex.core.processors.file import FileProcessor


class Keypoint:
    def __init__(self, x: float | int, y: float | int, visible: int) -> None:
        assert visible in [0, 1], "Keypoint visibility parameter must be one of [0, 1]."

        self.x = x
        self.y = y
        self.visible = visible

    def clip(self, *args) -> None:
        raise NotImplementedError(
            "Clip method must be implemented in subclasses of Keypoint."
        )


class RelativeKeypoint(Keypoint):
    """
    Keypoint with coordinates in [0, 1] relative to image size.
    """

    def clip(self) -> None:
        self.x = max(0, min(self.x, 1.0))
        self.y = max(0, min(self.y, 1.0))

    def to_absolute(self, img_width: int, img_height: int) -> "AbsoluteKeypoint":
        abs_x = int(self.x * img_width)
        abs_y = int(self.y * img_height)
        return AbsoluteKeypoint(abs_x, abs_y, self.visible)


class AbsoluteKeypoint(Keypoint):
    """
    Keypoint with coordinates in absolute pixel values.
    """

    def clip(self, img_width: int, img_height: int) -> None:
        self.x = max(0, min(self.x, img_width - 1))
        self.y = max(0, min(self.y, img_height - 1))

    def to_relative(self, img_width: int, img_height: int) -> "RelativeKeypoint":
        rel_x = self.x / img_width
        rel_y = self.y / img_height
        return RelativeKeypoint(rel_x, rel_y, self.visible)
