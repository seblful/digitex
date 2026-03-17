import numpy as np


class Converter:
    @staticmethod
    def point_to_polygon(
        point: list[float], img_width: int, img_height: int
    ) -> np.ndarray:
        arr = np.array(point)
        polygon = arr.reshape((-1, 2))
        polygon = polygon * np.array((img_width, img_height))

        return polygon

    @staticmethod
    def polygon_to_point(polygon: np.ndarray, img_width: int, img_height: int) -> list[float]:
        point = polygon / np.array((img_width, img_height))
        point = point.flatten().tolist()

        return point
