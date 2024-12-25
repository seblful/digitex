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


class OCRAnnsConverter(AnnsConverter):
    pass


class OCRBBOXAnnsConverter(OCRAnnsConverter):
    def __init__(self) -> None:
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
