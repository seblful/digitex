import os
import shutil
import random

import json
import yaml

import lmdb

from urllib.parse import unquote


class DatasetCreator():
    def __init__(self,
                 raw_dir: str,
                 dataset_dir: str,
                 train_split: float = 0.8) -> None:
        # Input paths
        self.raw_dir = raw_dir
        self.raw_images_dir = os.path.join(raw_dir, "images")

        self.data_json_path = os.path.join(raw_dir, 'data.json')
        self.chars_txt_path = os.path.join(raw_dir, "chars.txt")

        # Output paths
        self.dataset_dir = dataset_dir
        self.__setup_dataset_dirs()

        self.charset_yaml_path = os.path.join(raw_dir, "charset.yaml")
        self.gt_json_path = os.path.join(raw_dir, "gt.json")

        # Data split
        self.train_split = train_split
        self.val_split = 0.6 * (1 - self.train_split)
        self.test_split = 1 - self.train_split - self.val_split

        # Annotation creator
        self.annotation_creator = AnnotationCreator(raw_images_dir=self.raw_images_dir,
                                                    data_json_path=self.data_json_path,
                                                    chars_txt_path=self.chars_txt_path,
                                                    charset_yaml_path=self.charset_yaml_path,
                                                    gt_json_path=self.gt_json_path)

    def __setup_dataset_dirs(self) -> None:
        os.mkdir(self.dataset_dir)

        # Dataset dirs
        self.train_dir = os.path.join(self.dataset_dir, "train", "real")
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        self.dataset_dirs = [self.train_dir, self.val_dir, self.test_dir]

        # Create dirs
        for dir in self.dataset_dirs:
            os.makedirs(dir)

    @staticmethod
    def read_image(image_path) -> bytes:
        with open(image_path, 'rb') as image_file:
            bin_image = image_file.read()

        return bin_image

    @staticmethod
    def write_cache(env, cache) -> None:
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k, v)

    def __create_lmdb(self,
                      gt: dict[str, str],
                      set_dir: str) -> None:
        # Open lmdb database
        env = lmdb.open(set_dir, map_size=100000000)

        cache = {}
        counter = 1
        n_samples = len(gt)

        # Iterate through image and label
        for image_name, label in gt.items():
            image_path = os.path.join(self.raw_images_dir, image_name)
            bin_image = self.read_image(image_path)
            image_key = 'image-%09d'.encode() % counter
            label_key = 'label-%09d'.encode() % counter
            cache[image_key] = bin_image
            cache[label_key] = label.encode()

            # Write if counter
            if counter % 1000 == 0:
                self.write_cache(env, cache)
                cache = {}
            counter += 1

        # Write remain cache
        n_samples = counter - 1
        cache['num-samples'.encode()] = str(n_samples).encode()
        self.write_cache(env, cache)
        env.close()

    def __partitionate_data(self) -> None:
        # Load gt dict and shuffle
        gt_dict = self.annotation_creator.read_json(
            json_path=self.gt_json_path)
        keys = list(gt_dict.keys())
        random.shuffle(keys)
        gt_dict = {key: gt_dict[key] for key in keys}

        # Create train, validation and test gt dicts
        num_train = int(len(gt_dict) * self.train_split)
        num_val = int(len(gt_dict) * self.val_split)
        num_test = int(len(gt_dict) * self.test_split)

        train_gt = {key: gt_dict[key]
                    for key in list(gt_dict.keys())[:num_train]}
        val_gt = {key: gt_dict[key] for key in list(
            gt_dict.keys())[num_train:num_train+num_val]}
        test_gt = {key: gt_dict[key] for key in list(
            gt_dict.keys())[num_train+num_val:num_train+num_val+num_test]}

        # Create lmdb database for each set
        for gt, set_dir in zip((train_gt, val_gt, test_gt), (self.dataset_dirs)):
            self.__create_lmdb(gt=gt,
                               set_dir=set_dir)

        # Copy charset
        shutil.copy(self.charset_yaml_path, self.dataset_dir)

    def create_dataset(self) -> None:
        # Create annotations
        print("Annotations are creating...")
        self.annotation_creator.create_annotations()

        # Create dataset
        print("Data is partitioning...")
        self.__partitionate_data()


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
