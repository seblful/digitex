import json
import yaml


class FileProcessor:
    @staticmethod
    def read_txt(txt_path) -> list[str]:
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            content = txt_file.readlines()

        return content

    @staticmethod
    def write_txt(txt_path: str, lines: list[str]) -> None:
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.writelines(lines)

        return None

    @staticmethod
    def read_json(json_path) -> dict:
        try:
            with open(json_path, "r", encoding="utf-8") as json_file:
                json_dict = json.load(json_file)

        except FileNotFoundError:
            json_dict = {}

        return json_dict

    @staticmethod
    def write_json(json_dict: dict, json_path: str, indent: int = 4) -> None:
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(json_dict, json_file, indent=indent, ensure_ascii=False)

        return None

    @staticmethod
    def write_yaml(yaml_path: str, data: dict, comment: str = None) -> None:
        with open(yaml_path, "w", encoding="utf-8") as yaml_file:
            if comment:
                yaml_file.write(comment)
            yaml.dump(data, yaml_file, default_flow_style=False, allow_unicode=True)

        return None
