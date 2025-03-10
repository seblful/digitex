import numpy as np
import cv2


class Converter:
    @staticmethod
    def xyxyxyxy_to_polygon(xyxyxyxy: list[float],
                            img_width: int,
                            img_height: int) -> np.ndarray:
        xyxyxyxy = np.array(xyxyxyxy)
        polygon = xyxyxyxy.reshape((-1, 2))
        polygon = polygon * np.array((img_width, img_height))

        return polygon

    @staticmethod
    def point_to_polygon(point: list[float],
                         img_width: int,
                         img_height: int) -> np.ndarray:
        point = np.array(point)
        polygon = point.reshape((-1, 2))
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

    @staticmethod
    def point_to_keypoint(point: list[float],
                          img_width: int,
                          img_height: int,
                          clip: bool = True) -> list[tuple[int]]:
        point = np.array(point)
        keypoint = point.reshape((-1, 2))
        keypoint = keypoint * np.array((img_width, img_height))
        keypoint = keypoint.astype(int)
        if clip:
            keypoint = np.clip(keypoint, [0, 0], [
                               img_width - 1, img_height - 1])
        keypoint = keypoint.tolist()
        keypoint = [tuple(p) for p in keypoint]

        return keypoint

    @staticmethod
    def keypoint_to_point(keypoint: list[tuple[int]],
                          img_width: int,
                          img_height: int) -> list[tuple[float]]:
        keypoint = np.array(keypoint) / np.array((img_width, img_height))
        point = keypoint.tolist()
        point = [tuple(p) for p in point]

        return point
