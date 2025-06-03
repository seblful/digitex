import os
from urllib.parse import unquote

from tqdm import tqdm

from digitex.core.processors.file import FileProcessor


class BaseKeypoint:
    def __init__(self, x: float | int, y: float | int, visible: int) -> None:
        assert visible in [0, 1], "Keypoint visibility parameter must be one of [0, 1]."

        self.x = x
        self.y = y
        self.visible = visible

    def clip(self, *args) -> None:
        raise NotImplementedError(
            "Clip method must be implemented in subclasses of Keypoint."
        )


class RelativeKeypoint(BaseKeypoint):
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


class AbsoluteKeypoint(BaseKeypoint):
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


class BaseKeypointsObject:
    def __init__(
        self, class_idx: int, keypoints: list[BaseKeypoint], num_keypoints: int
    ) -> None:
        self.class_idx = class_idx
        self.num_keypoints = num_keypoints
        self.keypoints = self.pad_keypoints(keypoints, num_keypoints)

    def pad_keypoints(self, *args) -> list[BaseKeypoint]:
        raise NotImplementedError(
            "pad_keypoints method must be implemented in subclasses of BaseKeypointsObject."
        )

    def get_vis_coords(self) -> list[tuple]:
        coords = []

        for kp in self.keypoints:
            if kp.visible == 1:
                coords.append((kp.x, kp.y))

        return coords


class RelativeKeypointsObject(BaseKeypointsObject):
    def pad_keypoints(
        self, keypoints: list[RelativeKeypoint], num_keypoints: int
    ) -> list[RelativeKeypoint]:
        keypoints = keypoints[:num_keypoints]

        if not keypoints:
            return []

        if len(keypoints) < num_keypoints:
            keypoints = keypoints + [RelativeKeypoint(0, 0, 0)] * (
                num_keypoints - len(keypoints)
            )

        return keypoints

    def to_absolute(
        self, img_width: int, img_height: int, clip: bool
    ) -> "AbsoluteKeypointsObject":
        abs_keypoints = []
        for kp in self.keypoints:
            abs_kp = kp.to_absolute(img_width, img_height)

            if clip:
                abs_kp.clip(img_width, img_height)
            abs_keypoints.append(abs_kp)

        return AbsoluteKeypointsObject(
            class_idx=self.class_idx,
            keypoints=abs_keypoints,
            num_keypoints=len(self.keypoints),
        )


class AbsoluteKeypointsObject(BaseKeypointsObject):
    def __init__(
        self, class_idx: int, keypoints: list[BaseKeypoint], num_keypoints: int
    ) -> None:
        super().__init__(class_idx, keypoints, num_keypoints)

        self.__bbox = None

    @property
    def bbox(self) -> list[float]:
        if self.__bbox is None:
            if self.class_idx is None:
                return

            vis_coords = self.get_vis_coords()
            if not vis_coords:
                return

            # Calculate min and max coordinates
            min_x = min(kp[0] for kp in vis_coords)
            max_x = max(kp[0] for kp in vis_coords)
            min_y = min(kp[1] for kp in vis_coords)
            max_y = max(kp[1] for kp in vis_coords)

            # Get quadrilateral bbox
            self.__bbox = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
            ]

        return self.__bbox

    def pad_keypoints(self, keypoints: list[AbsoluteKeypoint], num_keypoints: int):
        keypoints = keypoints[:num_keypoints]

        if not keypoints:
            return []

        if len(keypoints) < num_keypoints:
            keypoints = keypoints + [AbsoluteKeypoint(0, 0, 0)] * (
                num_keypoints - len(keypoints)
            )
        return keypoints

    def to_relative(
        self, img_width: int, img_height: int, clip: bool = False
    ) -> "RelativeKeypointsObject":
        rel_keypoints = []
        for kp in self.keypoints:
            rel_kp = kp.to_relative(img_width, img_height)

            if clip:
                rel_kp.clip()
            rel_keypoints.append(rel_kp)

        return RelativeKeypointsObject(
            class_idx=self.class_idx,
            keypoints=rel_keypoints,
            num_keypoints=len(self.keypoints),
        )


class AnnotationCreator:
    def __init__(
        self,
        data_json_path: str,
        anns_json_path: str,
        num_keypoints: int = 30,
    ) -> None:
        self.data_json_path = data_json_path
        self.anns_json_path = anns_json_path

        self.num_keypoints = num_keypoints

    def get_keypoints_obj(self, task: dict) -> BaseKeypointsObject:
        keypoints = []

        # Get points
        result = task["annotations"][0]["result"]

        for entry in result:
            value = entry["value"]
            x = value["x"] / 100
            y = value["y"] / 100
            # label = value["keypointlabels"][0]
            keypoint = RelativeKeypoint(x, y, 1)
            keypoints.append(keypoint)

        if not keypoints:
            return RelativeKeypointsObject(
                class_idx=None, keypoints=[], num_keypoints=self.num_keypoints
            )

        return RelativeKeypointsObject(
            class_idx=0,  # Assuming a single class for keypoints
            keypoints=keypoints,
            num_keypoints=self.num_keypoints,
        )

    def create_annotations(self) -> None:
        # Read json
        json_dict = FileProcessor.read_json(self.data_json_path)

        # Create empty dict to store formatted annotations
        anns_dict = {}

        # Iterate through json dict
        for task in tqdm(json_dict, desc="Creating keypoints annotations"):
            image_name = unquote(os.path.basename(task["data"]["img"]))

            keypoints_obj = self.get_keypoints_obj(task)
            coords = [(kp.x, kp.y, kp.visible) for kp in keypoints_obj.keypoints]

            anns_dict[image_name] = coords

        # Save annotation
        FileProcessor.write_json(json_dict=anns_dict, json_path=self.anns_json_path)
