"""Label handling utilities."""

import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


class LabelHandler:
    """Handler for reading and processing label annotations."""

    @staticmethod
    def _read_points(label_path: str | Path) -> dict[int, list[list[float]]]:
        points_dict: dict[int, list[list[float]]] = dict()

        with open(label_path, "r") as f:
            for line in f:
                data = line.strip().split()
                if not data:
                    continue

                class_idx = int(data[0])
                points = [float(point) for point in data[1:]]

                if class_idx not in points_dict:
                    points_dict[class_idx] = []
                points_dict[class_idx].append(points)

        return points_dict

    @staticmethod
    def _get_random_points(
        classes_dict: dict[int, str],
        points_dict: dict[int, list],
        target_classes: list[str],
    ) -> tuple[int, list[float]]:
        """Get random points for a target class.

        Args:
            classes_dict: Mapping of class indices to names.
            points_dict: Dictionary of points by class index.
            target_classes: List of class names to filter by.

        Returns:
            Tuple of (points_index, points) or (-1, []) if no match found.
        """
        import random

        points_dict_filtered = {
            k: points_dict[k]
            for k in points_dict
            if classes_dict.get(k) in target_classes
        }

        if not points_dict_filtered:
            return -1, []

        rand_class_idx = random.choice(list(points_dict_filtered.keys()))

        rand_points_idx = random.randint(0, len(points_dict_filtered[rand_class_idx]) - 1)
        rand_points = points_dict_filtered[rand_class_idx][rand_points_idx]

        return rand_points_idx, rand_points

    @staticmethod
    def get_random_label(
        image_name: str,
        labels_dir: str | Path,
    ) -> Tuple[str | None, str | None]:
        """Get the label file path for a given image.

        Args:
            image_name: Name of the image file.
            labels_dir: Directory containing label files.

        Returns:
            Tuple of (label_name, label_path) or (None, None) if not found.
        """
        labels_dir = Path(labels_dir)

        label_name = Path(image_name).stem + '.txt'
        label_path = labels_dir / label_name

        if not label_path.exists():
            return None, None

        return label_name, str(label_path)

    @staticmethod
    def points_to_abs_polygon(
        points: list[float],
        image_width: int,
        image_height: int,
    ) -> list[tuple[int, int]]:
        """Convert normalized points to absolute pixel coordinates.

        Args:
            points: Normalized points [x1, y1, x2, y2, ...] (0-1).
            image_width: Image width in pixels.
            image_height: Image height in pixels.

        Returns:
            List of (x, y) tuples with absolute pixel coordinates.

        Raises:
            ValueError: If points list has odd length.
        """
        if len(points) % 2 != 0:
            raise ValueError("Points list must contain an even number of values")

        point_pairs = list(zip(points[::2], points[1::2]))
        abs_points = [
            (int(x * image_width), int(y * image_height))
            for x, y in point_pairs
        ]

        return abs_points

    def get_points(
        self,
        image_name: str,
        labels_dir: str | Path,
        classes_dict: dict[int, str],
        target_classes: list[str],
    ) -> tuple[int, list[float]]:
        """Get points for a specific image and target classes.

        Args:
            image_name: Name of the image file.
            labels_dir: Directory containing label files.
            classes_dict: Mapping of class indices to names.
            target_classes: List of class names to filter by.

        Returns:
            Tuple of (points_index, points).

        Raises:
            FileNotFoundError: If the label file doesn't exist.
        """
        _, rand_label_path = self.get_random_label(
            image_name=image_name,
            labels_dir=labels_dir,
        )

        if rand_label_path is None:
            return -1, []

        points_dict = self._read_points(rand_label_path)

        rand_points_idx, rand_points = self._get_random_points(
            classes_dict=classes_dict,
            points_dict=points_dict,
            target_classes=target_classes,
        )

        return rand_points_idx, rand_points
