import os
import json

from urllib.parse import quote, unquote
from PIL import Image

import numpy as np
import cv2


class AnnsConverter:
    def __init__(self,
                 ls_upload_dir: str):
        self.ls_upload_dir = ls_upload_dir

        self.bbox_keys = ["x", "y", "width", "height", "rotation"]

    @staticmethod
    def read_json(json_path) -> dict:
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_dict = json.load(json_file)

        return json_dict

    @staticmethod
    def write_json(json_dicts: dict,
                   json_path: str,
                   indent: int = 4) -> None:
        with open(json_path, 'w', encoding="utf-8") as json_file:
            json.dump(json_dicts, json_file,
                      indent=indent,
                      ensure_ascii=False)

        return None

    def add_filename_index(self,
                           filename: str,
                           index: int):
        filename_split = os.path.splitext(filename)
        filename = f"{filename_split[0]}_{str(index)}{filename_split[1]}"

        return filename

    def create_local_path(self,
                          task_path: str) -> str:
        task_path = unquote(task_path)
        task_path = os.path.normpath(task_path)
        path_split = task_path.split(os.sep)
        task_path = "/".join(path_split[3:])

        local_path = os.path.join(self.ls_upload_dir, task_path)
        local_path = os.path.normpath(local_path)

        return local_path

    def create_task_path(self,
                         local_path: str,
                         project_num: str = None,
                         index: int = None) -> str:
        path_split = local_path.split(os.sep)

        # Change project_num
        if project_num is not None:
            path_split[-2] = str(project_num)

        filename = path_split[-1]

        # Add index to filename
        if index is not None:
            filename = self.add_filename_index(filename=filename,
                                               index=index)

        filename = quote(filename)
        task_path = f"/data/{path_split[-3]}/{path_split[-2]}/{filename}"

        return task_path


class OCRAnnsConverter(AnnsConverter):
    def __init__(self,
                 ls_upload_dir: str) -> None:
        super().__init__(ls_upload_dir)


class OCRBBOXAnnsConverter(OCRAnnsConverter):
    def __init__(self,
                 ls_upload_dir: str) -> None:
        super().__init__(ls_upload_dir)

        self.output_json_name = "bbox_data.json"

    def get_preds(self, task: dict) -> list[dict]:
        # Create empty list to results
        preds = [{"result": []}]

        result = task['annotations'][0]['result']

        for entry in result:
            if entry['type'] == 'textarea':
                output_entry = entry.copy()

                del output_entry['value']['text']

                output_entry['value']['rectanglelabels'] = ['text']

                output_entry['from_name'] = 'label'
                output_entry['to_name'] = 'image'
                output_entry['type'] = 'rectanglelabels'

                preds[0]['result'].append(output_entry)

        return preds

    def convert(self,
                input_json_path: str,
                output_dir: str) -> list[dict]:
        # Read json_path
        input_json_dicts = self.read_json(input_json_path)

        output_json_dicts = []

        # Iterating through tasks
        for task in input_json_dicts:
            anns_dict = {}

            anns_dict['data'] = task['data']

            predictions = self.get_preds(task)
            anns_dict['predictions'] = predictions

            output_json_dicts.append(anns_dict)

        # Write to json file
        output_json_path = os.path.join(output_dir, self.output_json_name)
        self.write_json(json_dicts=output_json_dicts,
                        json_path=output_json_path)

        return output_json_dicts
