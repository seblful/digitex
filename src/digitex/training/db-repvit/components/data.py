import os
import shutil

import math
import random

import hashlib
from urllib.parse import unquote

from digitex.core.processors.file import FileProcessor


class AnnotationCreator:
    def __init__(
        self, raw_images_dir: str, data_json_path: str, anns_json_path: str
    ) -> None:
        # Paths
        self.raw_images_dir = raw_images_dir
        self.data_json_path = data_json_path
        self.anns_json_path = anns_json_path

    def image_hash(self, image_name: str) -> str:
        image_path = os.path.join(self.raw_images_dir, image_name)
        with open(image_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()

        return hash

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

        for entry in result:
            # Image width and height
            image_width = entry["original_width"]
            image_height = entry["original_height"]

            # Extract annotation from bbox
            if {"x", "text"}.issubset(entry["value"].keys()):
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
