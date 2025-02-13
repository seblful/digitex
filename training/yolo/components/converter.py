import numpy as np
import cv2


class Converter:
    @staticmethod
    def xyxyxyxy_to_polygon(xyxyxyxy: list[float],
                            img_width: int,
                            img_height: int) -> np.ndarray:
        # TODO np.array, reshape, w and h
        xyxyxyxy = [xyxyxyxy[i] * (img_width if i % 2 == 0 else img_height)
                    for i in range(len(xyxyxyxy))]
        xyxyxyxy = np.array(xyxyxyxy)
        polygon = xyxyxyxy.reshape((-1, 2))

        return polygon

    @staticmethod
    def point_to_polygon(point: list[float],
                         img_width: int,
                         img_height: int) -> np.ndarray:
        # TODO np.array and then reshape((-1, 2))
        polygon = list(zip(point[::2], point[1::2]))
        polygon = np.array(polygon) * np.array((img_width, img_height))

        return polygon

    @staticmethod
    def polygon_to_xyxyxyxy(polygon: np.ndarray,
                            img_width: int,
                            img_height: int) -> list[float]:
        # TODO minus in points
        polygon = polygon / np.array((1, 1))
        polygon = polygon.astype(np.float32)

        rect = cv2.minAreaRect(polygon)
        xyxyxyxy = cv2.boxPoints(rect)
        xyxyxyxy = xyxyxyxy / np.array((img_width, img_height))
        xyxyxyxy = xyxyxyxy.flatten().tolist()

        assert len(xyxyxyxy) == 8, "Length of xyxyxyxy must be equal 8."

        return xyxyxyxy

    @staticmethod
    def polygon_to_point(polygon: np.ndarray,
                         img_width: int,
                         img_height: int) -> list[float]:
        point = polygon / np.array((img_width, img_height))
        point = point.flatten().tolist()

        return point
