"""Prediction result classes for various prediction types."""


class PredictionResult:
    """Base class for prediction results."""

    def __init__(self, id2label: dict[int, str]) -> None:
        """Initialize a prediction result.

        Args:
            id2label: Dictionary mapping class IDs to label names.
        """
        self.id2label = id2label
        self.__label2id: dict[str, int] | None = None

    @property
    def label2id(self) -> dict[str, int]:
        """Get label to ID mapping.

        Returns:
            Dictionary mapping label names to class IDs.
        """
        if self.__label2id is None:
            self.__label2id = {v: k for k, v in self.id2label.items()}

        return self.__label2id


class DetectionPredictionResult(PredictionResult):
    """Prediction result for object detection tasks."""

    def __init__(
        self,
        ids: list[int],
        points: list[list[int]],
        id2label: dict[int, str],
    ) -> None:
        """Initialize a detection prediction result.

        Args:
            ids: List of class IDs for each detection.
            points: List of detection bounding boxes in xyxyxyxy format.
            id2label: Dictionary mapping class IDs to label names.

        Raises:
            ValueError: If points or ids are invalid.
        """
        if not points:
            raise ValueError("Points list cannot be empty")
        if len(points[0]) != 8:
            raise ValueError("Number of points for xyxyxyxy format must be equal 8.")
        if len(points) != len(ids):
            raise ValueError("Number of points must be equal to number of indexes.")

        super().__init__(id2label)
        self.ids = ids
        self.points = points

    @property
    def polygons(self) -> list[list[tuple[int, int]]]:
        """Get polygons from xyxyxyxy format.

        Returns:
            List of polygons as lists of (x, y) tuples.
        """
        polygons = []

        for point in self.points:
            polygon = list(zip(point[::2], point[1::2]))
            polygons.append(polygon)

        return polygons

    @property
    def id2points(self) -> dict[int, list[list[int]]]:
        """Get points grouped by class ID.

        Returns:
            Dictionary mapping class IDs to lists of point lists.
        """
        id_to_points: dict[int, list[list[int]]] = {}

        for idx, point in zip(self.ids, self.points):
            if idx not in id_to_points:
                id_to_points[idx] = []
            id_to_points[idx].append(point)

        return id_to_points

    @property
    def id2polygons(self) -> dict[int, list[list[tuple[int, int]]]]:
        """Get polygons grouped by class ID.

        Returns:
            Dictionary mapping class IDs to lists of polygons.
        """
        id_to_polygons: dict[int, list[list[tuple[int, int]]]] = {}

        for idx, polygon in zip(self.ids, self.polygons):
            if idx not in id_to_polygons:
                id_to_polygons[idx] = []
            id_to_polygons[idx].append(polygon)

        return id_to_polygons


class SegmentationPredictionResult(PredictionResult):
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

        super().__init__(id2label)
        self.ids = ids
        self.polygons = polygons

    @property
    def id2polygons(self) -> dict[int, list[list[tuple[int, int]]]]:
        """Get polygons grouped by class ID.

        Returns:
            Dictionary mapping class IDs to lists of polygons.
        """
        id_to_polygons: dict[int, list[list[tuple[int, int]]]] = {}

        for idx, polygon in zip(self.ids, self.polygons):
            if idx not in id_to_polygons:
                id_to_polygons[idx] = []
            id_to_polygons[idx].append(polygon)

        return id_to_polygons


class RecognitionPredictionResult(PredictionResult):
    """Prediction result for OCR/recognition tasks."""

    def __init__(
        self,
        text: str,
        probability: float,
        id2label: dict[int, str],
    ) -> None:
        """Initialize a recognition prediction result.

        Args:
            text: Recognized text string.
            probability: Confidence score for the recognition.
            id2label: Dictionary mapping class IDs to label names.
        """
        super().__init__(id2label)

        self.text = text
        self.probability = round(probability, 5)
