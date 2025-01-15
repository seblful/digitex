import os
import shutil
import random

import json
import yaml

from urllib.parse import unquote


class AnnotationCreator:
    def __init__(self,
                 raw_images_dir: str,
                 chars_txt_path: str,
                 replaces_json_path: str,
                 charset_yaml_path: str,
                 gt_json_path: str,
                 max_text_length: int) -> None:
        # Paths
        self.raw_images_dir = raw_images_dir
        self.chars_txt_path = chars_txt_path
        self.replaces_json_path = replaces_json_path
        self.charset_yaml_path = charset_yaml_path
        self.gt_json_path = gt_json_path

        self.max_text_length = max_text_length

        self.__charset = None
        self.__replaces_table = None

    @ property
    def charset(self) -> set[str]:
        if self.__charset is None:
            charset = self.read_txt(self.chars_txt_path)[0].strip()
            self.__charset = set(charset)

        return self.__charset

    @ property
    def replaces_table(self) -> dict:
        if self.__replaces_table is None:
            replaces_dict = self.read_json(self.replaces_json_path)
            self.__replaces_table = str.maketrans(replaces_dict)

        return self.__replaces_table

    @ staticmethod
    def read_txt(txt_path) -> list[str]:
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            content = txt_file.readlines()

        return content

    @ staticmethod
    def write_txt(txt_path: str,
                  lines: list[str]) -> None:
        with open(txt_path, 'w', encoding="utf-8") as txt_file:
            txt_file.writelines(lines)

        return None

    @ staticmethod
    def read_json(json_path) -> dict:
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_dict = json.load(json_file)

        return json_dict

    @ staticmethod
    def write_json(json_dict: dict,
                   json_path: str,
                   indent: int = 4) -> None:
        with open(json_path, 'w', encoding="utf-8") as json_file:
            json.dump(json_dict, json_file,
                      indent=indent,
                      ensure_ascii=False)

        return None

    @ staticmethod
    def write_yaml(yaml_path: str, data: dict, comment: str = None) -> None:
        with open(yaml_path, 'w', encoding="utf-8") as yaml_file:
            if comment:
                yaml_file.write(comment)
            yaml.dump(data, yaml_file,
                      default_flow_style=False,
                      allow_unicode=True)

        return None

    def __get_text(self, task: dict) -> str:
        # Retrieve result and label
        result = task['annotations'][0]['result']
        text = result[0]['value']['text'][0]

        return text

    def __get_image_path(self, task: dict) -> str:
        image_path = unquote(os.path.basename(task["data"]["captioning"]))
        # image_path = os.path.join(self.raw_images_dir, image_name)

        return image_path

    def check_length(self, text: str):
        if len(text) > self.max_text_length:
            raise ValueError(f"Length of text {text} is greater than {
                             self.max_text_length}.")

    def check_chars(self, text: str) -> None:
        for char in text:
            if char not in self.charset:
                raise ValueError(f"Char {char} not in charset.")

        return None

    def create_gt(self, image_paths: list[str], texts: list[str]) -> None:
        # Empty list to store gt
        gt = {}

        # Iterating through texts, image_paths and write to txt
        for image_path, text in zip(image_paths, texts):
            gt[image_path] = text

        # Write gts to json
        self.write_json(gt, self.gt_json_path)

    def create_annotations_from_ls(self,
                                   data_json_path: str) -> None:
        # Read jsons
        json_dict = self.read_json(data_json_path)

        texts = []
        image_paths = []

        # Iterate through task
        for task in json_dict:
            image_path = self.__get_image_path(task)
            text = self.__get_text(task)

            # Replace chars that not in the charset
            text = text.translate(self.replaces_table)

            # Check text length, chars in charset
            self.check_length(text)
            self.check_chars(text)

            texts.append(text)
            image_paths.append(image_path)

        # Create GT labels
        self.create_gt(image_paths, texts)

    def create_annotations_from_synth(self,
                                      gt_txt_path: str) -> None:
        lines = self.read_txt(gt_txt_path)

        texts = []
        image_paths = []

        for line in lines:
            line = line.strip()
            image_path, text = line.split(maxsplit=1)
            image_path = os.path.relpath(image_path, start="images")

            # Replace chars that not in the charset
            text = text.translate(self.replaces_table)

            # Check text length, chars in charset
            self.check_length(text)
            self.check_chars(text)

            texts.append(text)
            image_paths.append(image_path)

        # Create GT labels
        self.create_gt(image_paths, texts)
