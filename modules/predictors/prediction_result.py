class PredictionResult:
    def __init__(self,
                 id2label: dict[int, str]) -> None:
        self.id2label = id2label
        self.__label2id = None

    @property
    def label2id(self) -> dict[str, int]:
        if self.__label2id is None:
            self.__label2id = {v: k for k, v in self.id2label.items()}

        return self.__label2id


class DetectionPredictionResult(PredictionResult):
    def __init__(self,
                 ids: list[int],
                 points: list[list[int]],  # abs xyxyxyxy
                 id2label: dict[int, str]) -> None:

        assert len(
            points[0]) == 8, "Number of points for xyxyxyxy format must be equal 8."
        assert len(points) == len(
            ids), "Number of points must be equal to number of indexes."

        super().__init__(id2label)
        self.ids = ids
        self.points = points

    @property
    def polygons(self) -> list[list[tuple[int, int]]]:
        polygons = []

        for point in self.points:
            polygon = list(zip(point[::2], point[1::2]))
            polygons.append(polygon)

        return polygons

    @property
    def id2points(self) -> dict[int, list[int]]:
        id_to_points = {}

        for idx, point in zip(self.ids, self.points):
            if idx not in id_to_points:
                id_to_points[idx] = []
            id_to_points[idx].append(point)

        return id_to_points

    @property
    def id2polygons(self) -> dict[int, list[tuple[int, int]]]:
        id_to_polygons = {}

        for idx, polygon in zip(self.ids, self.polygons):
            if idx not in id_to_polygons:
                id_to_polygons[idx] = []
            id_to_polygons[idx].append(polygon)

        return id_to_polygons


class SegmentationPredictionResult(PredictionResult):
    def __init__(self,
                 ids: list[int],
                 polygons: list[list[tuple[int, int]]],  # abs [(xy), (xy)] xyn
                 id2label: dict[int, str]) -> None:
        assert len(polygons) == len(
            ids), "Number of points must be equal to number of indexes."

        super().__init__(id2label)
        self.ids = ids
        self.polygons = polygons

    @property
    def id2polygons(self) -> dict[int, list[tuple[int, int]]]:
        id_to_polygons = {}

        for idx, polygon in zip(self.ids, self.polygons):
            if idx not in id_to_polygons:
                id_to_polygons[idx] = []
            id_to_polygons[idx].append(polygon)

        return id_to_polygons
