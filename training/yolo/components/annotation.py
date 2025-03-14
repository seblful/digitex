import os
from urllib.parse import unquote

from tqdm import tqdm

from modules.processors import FileProcessor


class Keypoint:
    def __init__(self,
                 x: float,
                 y: float,
                 visible: int) -> None:
        assert visible in [
            0, 1], "Keypoint visibility parameter must be one of [0, 1]."

        self.x = x
        self.y = y
        self.visible = visible


class KeypointsObject:
    def __init__(self,
                 class_idx: int,
                 keypoints: list[Keypoint],
                 num_keypoints: int) -> None:
        self.class_idx = class_idx
        self.keypoints = self.pad_keypoints(keypoints, num_keypoints)

        self.bbox_offset = 1.05

        self.calc_props()

    def pad_keypoints(self, keypoints: list[Keypoint], num_keypoints: int) -> list[Keypoint]:
        keypoints = keypoints[:num_keypoints]

        if len(keypoints) < num_keypoints:
            keypoints = keypoints + \
                [Keypoint(0, 0, 0)] * (num_keypoints - len(keypoints))

        return keypoints

    def calc_props(self) -> None:
        if self.class_idx is None:
            self.width = 0
            self.height = 0
            self.center = (0, 0)

            return

        # Find min and max coordinates
        min_x = min(k.x for k in self.keypoints if k.visible == 1)
        max_x = max(k.x for k in self.keypoints if k.visible == 1)
        min_y = min(k.y for k in self.keypoints if k.visible == 1)
        max_y = max(k.y for k in self.keypoints if k.visible == 1)

        # Calculate width and height
        self.width = min((max_x - min_x) * self.bbox_offset, 1.0)
        self.height = min((max_y - min_y) * self.bbox_offset, 1.0)

        # Calculate center coordinates
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        self.center = (center_x, center_y)

    def to_string(self) -> str:
        if self.class_idx is None:
            return ""

        # Get all props
        props = [self.class_idx, self.center[0],
                 self.center[1], self.width, self.height]
        props = [str(prop) for prop in props]

        # Get all coords
        coords = []
        for keypoint in self.keypoints:
            coords.append(keypoint.x)
            coords.append(keypoint.y)
            coords.append(keypoint.visible)
        coords = [str(coord) for coord in coords]

        keypoints_str = " ".join(props + coords)

        return keypoints_str


class AnnotationCreator:
    def __init__(self,
                 raw_dir: str,
                 id2label: dict[int, str],
                 label2id: dict[str, int],
                 num_keypoints: int = 30) -> None:
        self.raw_dir = raw_dir
        self.data_json_path = os.path.join(raw_dir, "data.json")
        self.classes_path = os.path.join(raw_dir, 'classes.txt')
        self.__labels_dir = None

        self.id2label = id2label
        self.label2id = label2id

        self.num_keypoints = num_keypoints

    @property
    def labels_dir(self) -> str:
        if self.__labels_dir is None:
            self.__labels_dir = os.path.join(self.raw_dir, "labels")
            os.mkdir(self.__labels_dir)

        return self.__labels_dir

    def get_keypoints_obj(self, task: dict) -> tuple:
        keypoints = []

        # Get points
        result = task["annotations"][0]["result"]

        for entry in result:
            value = entry["value"]
            x = value["x"] / 100
            y = value["y"] / 100
            label = value["keypointlabels"][0]
            keypoint = Keypoint(x, y, 1)
            keypoints.append(keypoint)

        if not keypoints:
            return KeypointsObject(class_idx=None,
                                   keypoints=[],
                                   num_keypoints=self.num_keypoints)

        return KeypointsObject(class_idx=self.label2id[label],
                               keypoints=keypoints,
                               num_keypoints=self.num_keypoints)

    def create_keypoints(self) -> None:
        # Read json
        json_dict = FileProcessor.read_json(self.data_json_path)

        # Iterate through json dict
        for task in tqdm(json_dict, desc="Creating keypoints annotations"):
            image_name = unquote(os.path.basename(task["data"]["img"]))

            keypoints_obj = self.get_keypoints_obj(task)
            keypoints_str = keypoints_obj.to_string()

            # Write txt
            txt_name = os.path.splitext(image_name)[0] + ".txt"
            txt_path = os.path.join(self.labels_dir, txt_name)
            FileProcessor.write_txt(txt_path, lines=[keypoints_str])
