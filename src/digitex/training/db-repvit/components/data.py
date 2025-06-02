import os
import shutil
import random
import json
import math
import re

from urllib.parse import unquote

from digitex.core.processors.file import FileProcessor


class DatasetCreator:
    def __init__(
        self, raw_dir: str, dataset_dir: str, train_split: float = 0.8
    ) -> None:
        # Paths
        self.raw_dir = raw_dir
        self.raw_images_dir = os.path.join(raw_dir, "images")
        self.anns_json_path = os.path.join(raw_dir, "anns.json")

        self.dataset_dir = dataset_dir
        self._setup_dataset_dirs()

        self.data_json_path = os.path.join(raw_dir, "data.json")

        # Data split
        self.train_split = train_split
        self.val_split = 1 - self.train_split

        # Annotation creator
        self.annotation_creator = AnnotationCreator(
            raw_images_dir=self.raw_images_dir,
            data_json_path=self.data_json_path,
            anns_json_path=self.anns_json_path,
        )

    def _setup_dataset_dirs(self) -> None:
        os.mkdir(self.dataset_dir)

        # Train dirs
        self.train_dir = os.path.join(self.dataset_dir, "train")
        train_images_dir = os.path.join(self.train_dir, "images")
        os.makedirs(train_images_dir)

        # Val dirs
        self.val_dir = os.path.join(self.dataset_dir, "val")
        val_images_dir = os.path.join(self.val_dir, "images")
        os.makedirs(val_images_dir)

    def _copy_data(self, listdir: list[str], set_dir: str, anns_dict: dict) -> None:
        # Create empty dict to store formatted annotations
        set_anns_dict = {}

        # Copy image
        for image_name in listdir:
            shutil.copyfile(
                os.path.join(self.raw_images_dir, image_name),
                os.path.join(set_dir, "images", image_name),
            )
            set_anns_dict[image_name] = anns_dict[image_name]

        # Convert annotations to strings
        lines = []
        for k, v in set_anns_dict.items():
            path = os.path.join("images", k)
            ann = json.dumps(v, ensure_ascii=False)

            lines.append(path + "\t" + ann + "\n")

        label_path = os.path.join(set_dir, "labels.txt")
        FileProcessor.write_txt(label_path, lines)

    def _partitionate_data(self) -> None:
        # Images listdir and shuffle
        images_listdir = os.listdir(self.raw_images_dir)
        random.shuffle(images_listdir)

        # Create train and validation listdirs
        num_train = int(len(images_listdir) * self.train_split)
        num_val = int(len(images_listdir) * self.val_split)
        train_listdir = images_listdir[:num_train]
        val_listdir = images_listdir[num_train : num_train + num_val]

        # Load anns dict
        anns_dict = FileProcessor.read_json(json_path=self.anns_json_path)

        # Copy the images to folders and create annotation file
        for listdir, set_dir in zip(
            (train_listdir, val_listdir), (self.train_dir, self.val_dir)
        ):
            self._copy_data(listdir=listdir, set_dir=set_dir, anns_dict=anns_dict)

    def create_dataset(self) -> None:
        # Create annotations
        print("Annotations are creating...")
        self.annotation_creator.create_annotations()

        # Create dataset
        print("Data is partitioning...")
        self._partitionate_data()


class AnnotationCreator:
    def __init__(
        self, raw_images_dir: str, data_json_path: str, anns_json_path: str
    ) -> None:
        # Paths
        self.raw_images_dir = raw_images_dir
        self.data_json_path = data_json_path
        self.anns_json_path = anns_json_path

    @staticmethod
    def bbox_to_points(
        bbox: dict, image_width: int, image_height: int
    ) -> list[tuple[int, int]]:
        # Convert relative coordinates to absolute
        x = (bbox["x"] / 100) * image_width
        y = (bbox["y"] / 100) * image_height
        width = (bbox["width"] / 100) * image_width
        height = (bbox["height"] / 100) * image_height

        # Define the vertices of the rectangle (unrotated)
        vertices = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]

        # Convert rotation to radians
        angle_rad = math.radians(bbox["rotation"])

        # Define the rotation matrix
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        # Rotate each vertex around the top-left corner of the bounding box
        points = []
        for vx, vy in vertices:
            rx = x + (vx - x) * cos_theta - (vy - y) * sin_theta
            ry = y + (vx - x) * sin_theta + (vy - y) * cos_theta
            points.append((int(rx), int(ry)))

        return points

    def _get_annotation(self, task) -> list[dict]:
        # Create empty array to store data
        annotation = []

        # Retrieve result
        result = task["annotations"][0]["result"]

        # If no result
        if not result:
            return []

        # If result
        for entry in result:
            # Image width and height
            image_width = entry["original_width"]
            image_height = entry["original_height"]

            # Extract annotation from bbox
            if entry["from_name"] == "transcription":
                ann = {}
                ann["transcription"] = entry["value"]["text"][0]
                bbox = {k: v for k, v in entry["value"].items() if k != "text"}

                points = self.bbox_to_points(
                    bbox=bbox, image_width=image_width, image_height=image_height
                )
                ann["points"] = points

                annotation.append(ann)

        return annotation

    def create_annotations(self) -> None:
        # Read json data
        json_dict = FileProcessor.read_json(self.data_json_path)

        # Create empty dict to store formatted annotations
        anns_dict = {}

        # Iterating through tasks
        for task in json_dict:
            # Get annotation
            ann = self._get_annotation(task)

            # Fill ann dict
            image_name = unquote(os.path.basename(task["data"]["ocr"]))
            anns_dict[image_name] = ann

        # Save annotation
        FileProcessor.write_json(json_dict=anns_dict, json_path=self.anns_json_path)


class DataChecker:
    def __init__(self, raw_dir: str) -> None:
        self.data_json_path = os.path.join(raw_dir, "data.json")

        # Define regex patterns
        self.patterns = {
            "B": r'[^"]*B[^"]*',
            "A": r'[^"]*A[^"]*',
            "З": r'[^"]*З[^"]*',
            "О": r'[^"]*О[^"]*',
        }

    def _print_matches(self, task_id: int, char: str, text: str) -> None:
        print(f"Task ID: {task_id}, Found '{char}' in {text}")

    def _print_values(self, task_id: int, name: str, value: int, text: str) -> None:
        print(f"Task ID: {task_id}, Found {name}={value:.3e} in '{text}'")

    def check_text(self, task_id: int, entry: dict) -> None:
        transcription = entry["value"]["text"][0]

        for char, pattern in self.patterns.items():
            if re.search(pattern, transcription):
                self._print_matches(task_id, char, transcription)

    def check_bbox(self, task_id: int, entry: dict) -> None:
        values = entry["value"]
        text = values["text"][0]
        values = {name: value for name, value in values.items() if name != "text"}

        for name, value in values.items():
            if value < 0:
                self._print_values(task_id, name, value, text)

    def check(self) -> None:
        # Read json data
        json_dict = FileProcessor.read_json(self.data_json_path)

        # Iterate through tasks
        for task in json_dict:
            task_id = task.get("id")

            result = task["annotations"][0]["result"]

            for entry in result:
                if entry["from_name"] == "transcription":
                    self.check_text(task_id, entry)
                    self.check_bbox(task_id, entry)
