import os
import json


class AnnsConverter:
    def __init__(self):
        pass

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

    def ocr_to_bbox(self,
                    json_path: str):
        # Read json_path
        json_dict = self.read_json(json_path)

    def ocr_to_caption(self):
        pass
