import os
import shutil
import random

import json
import yaml

import lmdb

from urllib.parse import unquote


class AnnotationCreator:
    def __init__(self,
                 raw_images_dir: str,
                 data_json_path: str,
                 chars_txt_path: str,
                 charset_yaml_path: str,
                 gt_json_path: str) -> None:
        # Paths
        self.raw_images_dir = raw_images_dir
        self.data_json_path = data_json_path
        self.chars_txt_path = chars_txt_path
        self.charset_yaml_path = charset_yaml_path
        self.gt_json_path = gt_json_path

    @staticmethod
    def read_txt(txt_path) -> list[str]:
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            content = txt_file.read().strip("\n")

        return content

    @staticmethod
    def write_json(json_dict: dict,
                   json_path: str,
                   indent: int = 4) -> None:
        with open(json_path, 'w', encoding="utf-8") as json_file:
            json.dump(json_dict, json_file,
                      indent=indent,
                      ensure_ascii=False)

        return None

    @staticmethod
    def read_json(json_path) -> dict:
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_dict = json.load(json_file)

        return json_dict

    @staticmethod
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

    def create_charset(self, texts: list[str]) -> None:

        charset = set()
        for text in texts:
            charset.update(set(text))

        # Load and update with predefined chars
        chars = self.read_txt(self.chars_txt_path)
        charset.update(set(chars))
        charlist = sorted(list(charset))
        charstring = "".join(charlist)

        # Fill dict
        charset_dict = {"model": {
            "charset_train": charstring,
            "charset_test": charstring,
        }}

        # Write to yaml
        self.write_yaml(self.charset_yaml_path, charset_dict,
                        comment="# @package _global_\n")

        return None

    def create_gt(self, image_paths: list[str], texts: list[str]) -> None:
        # Empty list to store gt
        gt = {}

        # Iterating through texts, image_paths and write to txt
        for image_path, text in zip(image_paths, texts):
            gt[image_path] = text

        # Write gts to json
        self.write_json(gt, self.gt_json_path)

    def create_annotations(self) -> None:
        # Read data json
        json_dict = self.read_json(self.data_json_path)

        texts = []
        image_paths = []

        for task in json_dict:
            image_path = self.__get_image_path(task)
            text = self.__get_text(task)

            image_paths.append(image_path)
            texts.append(text)

        # Create charset and GT labels
        self.create_charset(texts)
        self.create_gt(image_paths, texts)
