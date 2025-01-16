import os
import shutil
import random

import json
import yaml

from urllib.parse import unquote


class DatasetCreator():
    def __init__(self,
                 raw_dir: str,
                 dataset_dir: str,
                 train_split: float = 0.8,
                 max_text_length=25) -> None:
        # Input paths
        self.raw_dir = raw_dir
        self.raw_images_dir = os.path.join(raw_dir, "images")
        self.chars_txt_path = os.path.join(raw_dir, "chars.txt")
        self.replaces_json_path = os.path.join(
            raw_dir, "replaces.json")

        # Output paths
        self.dataset_dir = dataset_dir
        self.charset_txt_path = os.path.join(dataset_dir, "charset.txt")
        self.gt_json_path = os.path.join(raw_dir, "gt.json")
        

        # Data split
        self.__setup_splits(train_split)

        # Sources
        self.sources = ["ls", "synth"]

        # Annotation creator
        self.annotation_creator = AnnotationCreator(raw_images_dir=self.raw_images_dir,
                                                    chars_txt_path=self.chars_txt_path,
                                                    replaces_json_path=self.replaces_json_path,
                                                    charset_txt_path = self.charset_txt_path,
                                                    gt_json_path=self.gt_json_path,
                                                    max_text_length=max_text_length)

    def __setup_splits(self, train_split: float) -> None:
        self.train_split = train_split
        self.val_split = 1 - self.train_split

    def __setup_dataset_dirs(self) -> None:
        os.mkdir(self.dataset_dir)

        # Dataset dirs
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.dataset_dirs = (self.train_dir, self.val_dir)

        # Create dirs
        for dir in self.dataset_dirs:
            image_dir = os.path.join(dir, "images")
            os.makedirs(image_dir)

    @staticmethod
    def sort_gt_key(key) -> int:
        filepath = key[0]
        basename = os.path.basename(filepath)
        filename = os.path.splitext(basename)[0]
        try:
            sort_key = int(filename)

        except ValueError:
            sort_key = filename

        return sort_key

    def __copy_data(self,
                    gt: dict[str, str],
                    set_dir: str) -> None:
        # Create list to store gt lines for txt gt
        gt_lines = []

        # Sort dict
        gt = dict(sorted(gt.items(), key=self.sort_gt_key))

        for image_path, text in gt.items():
            image_name = os.path.basename(image_path)
            shutil.copyfile(os.path.join(self.raw_images_dir, image_path),
                            os.path.join(set_dir, "images", image_name))

            # Create line and append it to the gt_lines
            gt_line = f"{image_name}\t{text}\n"
            gt_lines.append(gt_line)

        # Write set gt to txt
        gt_txt_path = os.path.join(set_dir, "gt.txt")
        self.annotation_creator.write_txt(txt_path=gt_txt_path,
                                          lines=gt_lines)

    def __partitionate_data(self) -> None:
        # Load gt dict and shuffle
        gt_dict = self.annotation_creator.read_json(
            json_path=self.gt_json_path)
        keys = list(gt_dict.keys())
        random.shuffle(keys)
        gt_dict = {key: gt_dict[key] for key in keys}

        # Create train and validation gt dicts
        num_train = int(len(gt_dict) * self.train_split)
        num_val = int(len(gt_dict) * self.val_split)

        train_gt = {key: gt_dict[key]
                    for key in list(gt_dict.keys())[:num_train]}
        val_gt = {key: gt_dict[key] for key in list(
            gt_dict.keys())[num_train:num_train+num_val]}

        # Create lmdb database for each set
        for gt, set_dir in zip((train_gt, val_gt), self.dataset_dirs):
            self.__copy_data(gt=gt,
                             set_dir=set_dir)

    def create_dataset(self, source: str) -> None:
        # Assert if right source
        assert source in self.sources, f"Source of raw images must be one of {
            self.sources}."

        # Create dataset dirs
        self.__setup_dataset_dirs()

        # Create annotations
        print("Annotations are creating...")
        if source == "ls":
            data_json_path = os.path.join(self.raw_dir, 'data.json')
            self.annotation_creator.create_annotations_from_ls(
                data_json_path=data_json_path)
        else:
            gt_txt_path = os.path.join(self.raw_dir, "gt.txt")
            self.annotation_creator.create_annotations_from_synth(
                gt_txt_path=gt_txt_path)

        # Create dataset
        print("Data is partitioning...")
        self.__partitionate_data()


class AnnotationCreator:
    def __init__(self,
                 raw_images_dir: str,
                 chars_txt_path: str,
                 replaces_json_path: str,
                 charset_txt_path:str,
                 gt_json_path: str,
                 max_text_length: int) -> None:
        # Paths
        self.raw_images_dir = raw_images_dir
        self.chars_txt_path = chars_txt_path
        self.replaces_json_path = replaces_json_path
        
        self.charset_txt_path = charset_txt_path
        self.gt_json_path = gt_json_path

        self.max_text_length = max_text_length

        self.__charset = None
        self.__replaces_table = None

    @property
    def charset(self) -> set[str]:
        if self.__charset is None:
            charset = self.read_txt(self.chars_txt_path)[0].strip()
            self.__charset = set(charset)

        return self.__charset

    @property
    def replaces_table(self) -> dict:
        if self.__replaces_table is None:
            replaces_dict = self.read_json(self.replaces_json_path)
            self.__replaces_table = str.maketrans(replaces_dict)

        return self.__replaces_table

    @staticmethod
    def read_txt(txt_path) -> list[str]:
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            content = txt_file.readlines()

        return content

    @staticmethod
    def write_txt(txt_path: str,
                  lines: list[str]) -> None:
        with open(txt_path, 'w', encoding="utf-8") as txt_file:
            txt_file.writelines(lines)

        return None

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
        
    def create_charset(self) -> None:
        chars = sorted(self.charset)
        
        # Create lines with char and new line
        charlines = []
        for char in chars:
            charline = char + "\n"
            charlines.append(charline)
        
        # Write lines to txt    
        self.write_txt(self.charset_txt_path, charlines)
        
        return None

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
        
        # Create charset
        self.create_charset()

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
        
        # Create charset
        self.create_charset()
