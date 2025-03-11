import os
from urllib.parse import unquote

from tqdm import tqdm

from modules.processors import FileProcessor


class AnnotationCreator:
    def __init__(self,
                 raw_dir) -> None:
        self.raw_dir = raw_dir
        self.data_json_path = os.path.join(raw_dir, "data.json")
        self.__labels_dir = None

    @property
    def labels_dir(self) -> str:
        if self.__labels_dir is None:
            self.__labels_dir = os.path.join(self.raw_dir, "labels")
            os.mkdir(self.__labels_dir)

        return self.__labels_dir

    @staticmethod
    def get_points_props(points: list[tuple[float]],
                         offset: float = 1.05) -> tuple[tuple[float, float], float, float]:
        # Find min and max coordinates
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)

        # Calculate width and height
        width = min((max_x - min_x) * offset, 1.0)
        height = min((max_y - min_y) * offset, 1.0)

        # Calculate center coordinates
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        return (center_x, center_y), width, height

    def get_keypoints(self, task: dict) -> tuple:
        points = []

        # Get points
        result = task["annotations"][0]["result"]

        for entry in result:
            value = entry["value"]
            x = value["x"] / 100
            y = value["y"] / 100
            points.append((x, y))

        if not points:
            return (), None, None, []

        # Calculate objet properties
        center, width, height = self.get_points_props(points)

        return center, width, height, points

    def create_keypoints_str(self,
                             center: tuple[float],
                             width: float,
                             height: float,
                             points: list[tuple[float]]) -> str:
        if not points:
            return ""

        points = [str(coord) for point in points for coord in point]
        points_str = " ".join(points)

        # TODO multiclass support
        keypoints_str = f"{0} {str(center[0])} {str(center[1])} {width} {height} {points_str}"

        return keypoints_str

    def create_keypoints_anns(self) -> None:
        # Read json
        json_dict = FileProcessor.read_json(self.data_json_path)

        # Iterate through json dict
        for task in tqdm(json_dict, desc="Creating keypoints annotations"):
            image_name = unquote(os.path.basename(task["data"]["img"]))

            center, width, height, points = self.get_keypoints(task)

            keypoints_str = self.create_keypoints_str(
                center, width, height, points)

            # Write txt
            txt_name = os.path.splitext(image_name)[0] + ".txt"
            txt_path = os.path.join(self.labels_dir, txt_name)
            FileProcessor.write_txt(txt_path, lines=[keypoints_str])
