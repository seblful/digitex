"""Prediction result for segmentation tasks."""


class SegmentationPredictionResult:
    """Prediction result for segmentation tasks."""

    def __init__(
        self,
        ids: list[int],
        polygons: list[list[tuple[int, int]]],
        id2label: dict[int, str],
    ) -> None:
        """Initialize a segmentation prediction result.

        Args:
            ids: List of class IDs for each segmentation mask.
            polygons: List of segmentation polygons as lists of (x, y) tuples.
            id2label: Dictionary mapping class IDs to label names.

        Raises:
            ValueError: If polygons or ids are invalid.
        """
        if len(polygons) != len(ids):
            raise ValueError("Number of polygons must be equal to number of indexes.")

        self.ids = ids
        self.polygons = polygons
        self.id2label = id2label

    @property
    def id2polygons(self) -> dict[int, list[list[tuple[int, int]]]]:
        """Get polygons grouped by class ID.

        Returns:
            Dictionary mapping class IDs to lists of polygons.
        """
        id_to_polygons: dict[int, list[list[tuple[int, int]]]] = {}

        for idx, polygon in zip(self.ids, self.polygons, strict=False):
            if idx not in id_to_polygons:
                id_to_polygons[idx] = []
            id_to_polygons[idx].append(polygon)

        return id_to_polygons
