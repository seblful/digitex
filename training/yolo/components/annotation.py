import os
from urllib.parse import unquote

from tqdm import tqdm

from modules.processors import FileProcessor


class Keypoint:
    def __init__(self,
                 x: float | int,
                 y: float | int,
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
                 num_keypoints: int,
                 bbox_center: tuple[float | int] = None,
                 bbox_width: float | int = None,
                 bbox_height: float | int = None) -> None:
        self.class_idx = class_idx
        self.keypoints = self.pad_keypoints(keypoints, num_keypoints)

        self.bbox_center = bbox_center
        self.bbox_width = bbox_width
        self.bbox_height = bbox_height

        self.bbox_offset = 1.05

        if None in (self.bbox_center, self.bbox_width, self.bbox_height):
            self.calc_props()

    def pad_keypoints(self, keypoints: list[Keypoint], num_keypoints: int) -> list[Keypoint]:
        keypoints = keypoints[:num_keypoints]

        if len(keypoints) < num_keypoints:
            keypoints = keypoints + \
                [Keypoint(0, 0, 0)] * (num_keypoints - len(keypoints))

        return keypoints

    def calc_props(self) -> None:
        print("CALCULATING PROPS")
        if self.class_idx is None:
            self.bbox_center = (0, 0)
            self.bbox_width = 0
            self.bbox_height = 0

            return

        visible_kps = [kp for kp in self.keypoints if kp.visible == 1]
        if not visible_kps:
            self.bbox_center = (0, 0)
            self.bbox_width = 0
            self.bbox_height = 0
            return

        # Calculate min and max coordinates
        min_x = min(kp.x for kp in visible_kps)
        max_x = max(kp.x for kp in visible_kps)
        min_y = min(kp.y for kp in visible_kps)
        max_y = max(kp.y for kp in visible_kps)

        # Calculate
        self.bbox_center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        self.bbox_width = min((max_x - min_x) * self.bbox_offset, 1.0)
        self.bbox_height = min((max_y - min_y) * self.bbox_offset, 1.0)

    def to_relative(self,
                    img_width: int,
                    img_height: int,
                    clip: bool = False) -> 'KeypointsObject':
        # Convert coordinates
        rel_keypoints = []

        for kp in self.keypoints:
            rel_x = int(kp.x * img_width)
            rel_y = int(kp.y * img_height)

            if clip:
                rel_x = max(0, min(rel_x, img_width - 1))
                rel_y = max(0, min(rel_y, img_height - 1))

            rel_keypoints.append(Keypoint(rel_x, rel_y, kp.visible))

        # Convert propertirs
        center_x = int(self.bbox_center[0] * img_width)
        center_y = int(self.bbox_center[1] * img_height)
        bbox_center = (center_x, center_y)
        bbox_width = int(self.bbox_width * img_width)
        bbox_height = int(self.bbox_height * img_height)

        return KeypointsObject(class_idx=self.class_idx,
                               keypoints=rel_keypoints,
                               num_keypoints=len(self.keypoints),
                               bbox_center=bbox_center,
                               bbox_width=bbox_width,
                               bbox_height=bbox_height)

    def to_string(self) -> str:
        if self.class_idx is None:
            return ""

        # Get all props
        props = [self.class_idx, self.bbox_center[0],
                 self.bbox_center[1], self.bbox_width, self.bbox_height]
        coords = [coord for kp in self.keypoints for coord in (
            kp.x, kp.y, kp.visible)]

        keypoints_str = " ".join(map(str, props + coords))

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
