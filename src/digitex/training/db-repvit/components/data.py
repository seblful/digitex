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
    def bbox_to_polygon(
        bbox: dict, image_width: int, image_height: int
    ) -> list[tuple[int, int]]:
        x_rel, y_rel, width_rel, height_rel, rotation = (
            bbox["x"],
            bbox["y"],
            bbox["width"],
            bbox["height"],
            bbox["rotation"],
        )

        # Convert relative coordinates to absolute
        x = (x_rel / 100) * image_width
        y = (y_rel / 100) * image_height
        width = (width_rel / 100) * image_width
        height = (height_rel / 100) * image_height

        # Define the vertices of the rectangle (unrotated)
        vertices = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]

        # Convert rotation to radians
        angle_rad = math.radians(rotation)

        # Define the rotation matrix
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        # Rotate each vertex around the top-left corner of the bounding box
        rotated_polygon = []
        for vx, vy in vertices:
            rx = x + (vx - x) * cos_theta - (vy - y) * sin_theta
            ry = y + (vx - x) * sin_theta + (vy - y) * cos_theta
            rotated_polygon.append((int(rx), int(ry)))

        return rotated_polygon

    @staticmethod
    def polygon_to_abs(polygon, image_width: int, image_height: int) -> None:
        for i in range(len(polygon)):
            polygon[i][0] = int((polygon[i][0] / 100) * image_width)
            polygon[i][1] = int((polygon[i][1] / 100) * image_height)

        return None

    def _get_annotation(self, task) -> list[dict]:
        # Create empty array to store data
        annotation = []

        # Retrieve result
        result = task["annotations"][0]["result"]

        for entry in result:
            # Image width and height
            image_width = entry["original_width"]
            image_height = entry["original_height"]

            # Create empty dict to store transcription and points
            ann = {}
            # Extract points (polygon)
            value_keys = entry["value"].keys()

            # Extract annotation from points
            if {"points", "text"}.issubset(value_keys):
                ann["transcription"] = entry["value"]["text"][0]
                polygon = entry["value"]["points"]
                self.polygon_to_abs(
                    polygon=polygon, image_width=image_width, image_height=image_height
                )
                polygon = [tuple(points) for points in polygon]
                ann["points"] = polygon

                annotation.append(ann)

            # Extract annotation from bbox
            elif {"x", "text"}.issubset(value_keys):
                ann["transcription"] = entry["value"]["text"][0]
                bbox = {
                    k: v for k, v in entry["value"].items() if k != "rectanglelabels"
                }

                polygon = self.bbox_to_polygon(
                    bbox=bbox, image_width=image_width, image_height=image_height
                )
                ann["points"] = polygon

                annotation.append(ann)

        return annotation

    def create_annotations(self) -> None:
        # Read json_polygon_path
        json_dict = FileProcessor.read_json(self.data_json_path)

        # Create empty dict to store formatted annotations
        anns_dict = {}

        # Iterating through tasks
        for task in json_dict:
            # Get polygons and labels
            ann = self._get_annotation(task)

            # Fill ann dict
            image_name = unquote(os.path.basename(task["data"]["ocr"]))
            anns_dict[image_name] = ann

        # Save annotation
        FileProcessor.write_json(json_dict=anns_dict, json_path=self.anns_json_path)
