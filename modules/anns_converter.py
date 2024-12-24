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
    def write_json(json_dicts: dict,
                   json_path: str,
                   indent: int = 4) -> None:
        with open(json_path, 'w', encoding="utf-8") as json_file:
            json.dump(json_dicts, json_file,
                      indent=indent,
                      ensure_ascii=False)

        return None

    def get_bbox_preds(self, task: dict):
        # Create empty list to results
        preds = [{"result": []}]

        result = task['annotations'][0]['result']

        for entry in result:
            if entry['type'] == 'textarea':
                entry_copy = entry.copy()

                del entry_copy['value']['text']

                entry_copy['value']['rectanglelabels'] = ['text']

                entry_copy['from_name'] = 'label'
                entry_copy['to_name'] = 'image'
                entry_copy['type'] = 'rectanglelabels'

                preds[0]["result"].append(entry_copy)

        return preds

    def ocr_to_bbox(self,
                    ocr_json_path: str,
                    output_dir: str) -> list[dict]:
        # Read json_path
        ocr_json_dicts = self.read_json(ocr_json_path)

        bbox_json_dicts = []

    def ocr_to_caption(self):
        pass
