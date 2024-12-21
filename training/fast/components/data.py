import os
import shutil
import json
import random

import hashlib
from urllib.parse import unquote


class AnnotationCreator:
    def __init__(self,
                 raw_images_dir: str,
                 data_json_path: str,
                 anns_json_path: str) -> None:
        # Paths
        self.raw_images_dir = raw_images_dir
        self.data_json_path = data_json_path
        self.anns_json_path = anns_json_path

    @staticmethod
    def read_json(json_path) -> dict:
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_dict = json.load(json_file)

        return json_dict

    @staticmethod
    def write_json(json_dict: dict,
                   json_path: str,
                   indent: int = 4) -> None:
        with open(json_path, 'w', encoding="utf-8") as json_file:
            json.dump(json_dict, json_file,
                      indent=indent,
                      ensure_ascii=False)

        return None

    def image_hash(self,
                   image_name: str) -> str:
        image_path = os.path.join(self.raw_images_dir, image_name)
        with open(image_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()

        return hash

    @staticmethod
    def bbox_to_polygon(bbox: list[float],
                        image_width: int,
                        image_height: int) -> list[int]:

        x_rel, y_rel, width_rel, height_rel = bbox

        # Convert relative coordinates to absolute
        x = int((x_rel / 100) * image_width)
        y = int((y_rel / 100) * image_height)
        width = int((width_rel / 100) * image_width)
        height = int((height_rel / 100) * image_height)

        # Define the vertices of the polygon
        polygon = [(x, y), (x + width, y),
                   (x + width, y + height), (x, y + height)]

        return polygon

    @staticmethod
    def polygon_to_abs(polygon,
                       image_width: int,
                       image_height: int) -> None:

        for i in range(len(polygon)):
            polygon[i][0] = int((polygon[i][0] / 100) * image_width)
            polygon[i][1] = int((polygon[i][1] / 100) * image_height)

        return None

    def __get_polygons(self, task) -> tuple[list, int, int]:
        # Create empty array to store data
        polygons = []

        # Retrieve result
        result = task['annotations'][0]['result']

        for entry in result:
            if entry['type'] == 'textarea':
                # Width and height
                image_width = entry["original_width"]
                image_height = entry["original_height"]

                # Extract bbox and convert it to polygon with absolute coordinates
                if 'x' in entry['value'].keys():
                    bbox = [entry['value']['x'],
                            entry['value']['y'],
                            entry['value']['width'],
                            entry['value']['height']]

                    polygon = self.bbox_to_polygon(bbox=bbox,
                                                   image_width=image_width,
                                                   image_height=image_height)
                    polygons.append(polygon)

                # Extract polygon and convert it to polygon with absolute coordinates
                elif 'points' in entry['value'].keys():
                    polygon = entry['value']['points']
                    self.polygon_to_abs(polygon=polygon,
                                        image_width=image_width,
                                        image_height=image_height)
                    polygon = [tuple(points) for points in polygon]

                    polygons.append(polygon)

        return polygons, image_width, image_height

    def create_annotations(self) -> None:
        # Read json_polygon_path
        json_dict = self.read_json(self.data_json_path)

        # Create empty dict to store formatted annotations
        anns_dict = {}

        # Iterating through tasks
        for task in json_dict:

            # Get polygons and labels
            polygons, image_width, image_height = self.__get_polygons(
                task)

            # Fill ann dict
            image_name = unquote(os.path.basename(task["data"]["image"]))
            anns_dict[image_name] = {"img_dimensions": (image_width, image_height),
                                     "img_hash": self.image_hash(image_name),
                                     "polygons": polygons}

        # Save annotation
        self.write_json(json_dict=anns_dict,
                        json_path=self.anns_json_path)
