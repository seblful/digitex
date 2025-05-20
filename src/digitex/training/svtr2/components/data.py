import os
import shutil
import random
from urllib.parse import unquote
from tqdm import tqdm
from digitex.core.processors.file import FileProcessor
from abc import ABC, abstractmethod

import lmdb
import cv2
import numpy as np
import io
from PIL import Image


class BaseDatasetCreator(ABC):
    def __init__(
        self,
        raw_dir: str,
        dataset_dir: str,
        train_split: float = 0.8,
        max_text_length=31,
    ) -> None:
        self.raw_dir = raw_dir
        self.dataset_dir = dataset_dir

        self.chars_txt_path = os.path.join(raw_dir, "chars.txt")
        self.replaces_json_path = os.path.join(raw_dir, "replaces.json")
        self.charset_txt_path = os.path.join(dataset_dir, "charset.txt")
        self._charset = None

        self._setup_splits(train_split)
        self.sources = ["ls", "synth"]

        self.annotation_creator = AnnotationCreator(
            charset=self.charset,
            replaces_json_path=self.replaces_json_path,
            max_text_length=max_text_length,
        )

    def _setup_splits(self, train_split: float) -> None:
        self.train_split = train_split
        self.val_split = 1 - train_split

    @property
    def charset(self) -> set[str]:
        if self._charset is None:
            charset = FileProcessor.read_txt(self.chars_txt_path)[0].rstrip()
            self._charset = set(charset)
        return self._charset

    @staticmethod
    def shuffle_dict(d: dict) -> dict:
        keys = list(d.keys())
        random.shuffle(keys)
        return {key: d[key] for key in keys}

    @staticmethod
    def sort_anns_key(key) -> tuple:
        filepath = key[0]
        basename = os.path.basename(filepath)
        filename = os.path.splitext(basename)[0]
        try:
            return (True, int(filename))
        except ValueError:
            return (False, filename)

    def _create_charset(self) -> None:
        chars = sorted(self.charset)
        FileProcessor.write_txt(self.charset_txt_path, chars, newline=True)

    def _partitionate_data(
        self, anns_dict: dict[str, str], aug_anns_dict: dict[str, str] = None
    ):
        anns_dict = self.shuffle_dict(anns_dict)
        num_train = int(len(anns_dict) * self.train_split)
        train_keys = list(anns_dict.keys())[:num_train]
        val_keys = list(anns_dict.keys())[num_train:]

        train_anns_dict = {key: anns_dict[key] for key in train_keys}
        val_anns_dict = {key: anns_dict[key] for key in val_keys}

        if aug_anns_dict:
            train_anns_dict.update(aug_anns_dict)
            train_anns_dict = self.shuffle_dict(train_anns_dict)

        return train_anns_dict, val_anns_dict

    @abstractmethod
    def create_dataset(self, source: str, use_aug=False) -> None:
        pass


class SimpleDatasetCreator(BaseDatasetCreator):
    def _setup_dataset_dirs(self) -> None:
        os.mkdir(self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.dataset_dirs = (self.train_dir, self.val_dir)
        for dir in self.dataset_dirs:
            os.mkdir(dir)
            images_dir = os.path.join(dir, "images")
            os.mkdir(images_dir)

    def _copy_data(
        self, anns_dict: dict[str, str], set_dir: str, images_per_folder: int = 10000
    ) -> None:
        dir_name = os.path.basename(set_dir)
        images_dir = os.path.join(set_dir, "images")

        lines = []
        anns_dict = dict(sorted(anns_dict.items(), key=self.sort_anns_key))
        subfolder_idx = img_in_subfolder = 0

        for image_path, text in tqdm(
            anns_dict.items(), desc=f"Partitioning {dir_name} data"
        ):
            if img_in_subfolder >= images_per_folder:
                subfolder_idx += 1
                img_in_subfolder = 0
            subfolder_name = str(subfolder_idx)
            subfolder_path = os.path.join(images_dir, subfolder_name)

            if not os.path.exists(subfolder_path):
                os.mkdir(subfolder_path)

            src_image_path = os.path.join(self.raw_dir, image_path)
            image_basename = os.path.basename(image_path)
            dst_image_path = os.path.join("images", subfolder_name, image_basename)
            dst_image_path = os.path.normpath(dst_image_path).replace("\\", "/")
            shutil.copyfile(src_image_path, os.path.join(set_dir, dst_image_path))

            lines.append(f"{dst_image_path}\t{text}")
            img_in_subfolder += 1

        labels_txt_path = os.path.join(set_dir, "labels.txt")
        FileProcessor.write_txt(txt_path=labels_txt_path, lines=lines, newline=True)

    def create_dataset(self, source: str, use_aug=False) -> None:
        assert source in self.sources, (
            f"Source of raw images must be one of {self.sources}."
        )
        self._setup_dataset_dirs()
        if source == "ls":
            images_dir = "images"
            data_json_path = os.path.join(self.raw_dir, "data.json")
            anns_dict = self.annotation_creator.create_ls_anns(
                images_dir=images_dir, data_json_path=data_json_path
            )
        else:
            images_dir = "images"
            gt_txt_path = os.path.join(self.raw_dir, "gt.txt")
            anns_dict = self.annotation_creator.create_synth_anns(
                images_dir=images_dir, gt_txt_path=gt_txt_path
            )
        aug_anns_dict = None

        if use_aug:
            images_dir = "aug-images"
            gt_txt_path = os.path.join(self.raw_dir, "aug_gt.txt")
            aug_anns_dict = self.annotation_creator.create_synth_anns(
                images_dir=images_dir, gt_txt_path=gt_txt_path
            )
        train_anns_dict, val_anns_dict = self._partitionate_data(
            anns_dict=anns_dict, aug_anns_dict=aug_anns_dict
        )
        for an_dict, set_dir in zip(
            (train_anns_dict, val_anns_dict), self.dataset_dirs
        ):
            self._copy_data(anns_dict=an_dict, set_dir=set_dir)
        self._create_charset()


class LMDBDatasetCreator(BaseDatasetCreator):
    def _setup_dataset_dirs(self) -> None:
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.dataset_dirs = (self.train_dir, self.val_dir)
        for dir in self.dataset_dirs:
            os.makedirs(dir, exist_ok=True)

    def _dict_to_list(self, anns_dict: dict[str, str]) -> list:
        # Convert dict to list of [image_path, label]
        return [[os.path.join(self.raw_dir, k), v] for k, v in anns_dict.items()]

    def _calculate_map_size(self, data_list):
        total_size = 0
        for imagePath, _ in data_list:
            if os.path.exists(imagePath):
                total_size += os.path.getsize(imagePath)
        # Add 50% buffer and minimum 128MB to avoid map full error
        return max(int(total_size * 1.5), 128 * 1024 * 1024)

    def _write_lmdb(self, data_list, outputPath, checkValid=True) -> None:
        os.makedirs(outputPath, exist_ok=True)
        map_size = self._calculate_map_size(data_list)
        env = lmdb.open(outputPath, map_size=map_size)
        cache = {}
        cnt = 1
        for image_path, label in tqdm(
            data_list, desc=f"Partitioning {os.path.basename(outputPath)} data"
        ):
            try:
                with open(image_path, "rb") as f:
                    image_bin = f.read()
                    buf = io.BytesIO(image_bin)
                    w, h = Image.open(buf).size
                if checkValid:
                    if not self._is_image_valid(image_bin):
                        print("%s is not a valid image" % image_path)
                        continue
            except Exception:
                continue

            image_key = "image-%09d".encode() % cnt
            label_key = "label-%09d".encode() % cnt
            wh_key = "wh-%09d".encode() % cnt
            cache[image_key] = image_bin
            cache[label_key] = label.encode()
            cache[wh_key] = (str(w) + "_" + str(h)).encode()

            if cnt % 1000 == 0:
                self._write_cache(env, cache)
                cache = {}
            cnt += 1
        num_samples = cnt - 1
        cache["num-samples".encode()] = str(num_samples).encode()
        self._write_cache(env, cache)

    @staticmethod
    def _is_image_valid(imageBin):
        if imageBin is None:
            return False
        image_buf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            return False
        return True

    @staticmethod
    def _write_cache(env, cache):
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k, v)

    def create_dataset(self, source: str, use_aug=False) -> None:
        assert source in self.sources, (
            f"Source of raw images must be one of {self.sources}."
        )
        self._setup_dataset_dirs()
        if source == "ls":
            images_dir = "images"
            data_json_path = os.path.join(self.raw_dir, "data.json")
            anns_dict = self.annotation_creator.create_ls_anns(
                images_dir=images_dir, data_json_path=data_json_path
            )
        else:
            images_dir = "images"
            gt_txt_path = os.path.join(self.raw_dir, "gt.txt")
            anns_dict = self.annotation_creator.create_synth_anns(
                images_dir=images_dir, gt_txt_path=gt_txt_path
            )
        aug_anns_dict = None

        if use_aug:
            images_dir = "aug-images"
            gt_txt_path = os.path.join(self.raw_dir, "aug_gt.txt")
            aug_anns_dict = self.annotation_creator.create_synth_anns(
                images_dir=images_dir, gt_txt_path=gt_txt_path
            )
        train_anns_dict, val_anns_dict = self._partitionate_data(
            anns_dict=anns_dict, aug_anns_dict=aug_anns_dict
        )
        # Write LMDB for train and val
        for an_dict, set_dir in zip(
            (train_anns_dict, val_anns_dict), self.dataset_dirs
        ):
            data_list = self._dict_to_list(an_dict)
            self._write_lmdb(data_list, set_dir)
        self._create_charset()


class AnnotationCreator:
    def __init__(
        self, charset: set[str], replaces_json_path: str, max_text_length: int
    ) -> None:
        self.charset = charset
        self.replaces_json_path = replaces_json_path
        self.max_text_length = max_text_length
        self._replaces_table = None

    @property
    def replaces_table(self) -> dict:
        if self._replaces_table is None:
            replaces_dict = FileProcessor.read_json(self.replaces_json_path)
            self._replaces_table = str.maketrans(replaces_dict)
        return self._replaces_table

    def check_length(self, text: str) -> None:
        if len(text) > self.max_text_length:
            raise ValueError(
                f"Length of text {text} is greater than {self.max_text_length}."
            )

    def check_chars(self, text: str) -> None:
        for char in text:
            if char not in self.charset:
                raise ValueError(f"Char {char} not in charset.")

    def _create_anns(self, image_paths: list[str], texts: list[str]) -> dict[str, str]:
        return dict(zip(image_paths, texts))

    def _get_text(self, task: dict) -> str:
        return task["annotations"][0]["result"][0]["value"]["text"][0]

    def _get_image_path(self, task: dict, images_dir: str) -> str:
        image_path = unquote(os.path.basename(task["data"]["captioning"]))
        return os.path.join(images_dir, image_path)

    def create_ls_anns(self, images_dir: str, data_json_path: str) -> dict[str, str]:
        json_dict = FileProcessor.read_json(data_json_path)
        texts, image_paths = [], []

        for task in tqdm(json_dict, desc="Creating annotations from Label Studio"):
            image_path = self._get_image_path(task, images_dir)

            text = self._get_text(task).translate(self.replaces_table)
            self.check_length(text)
            self.check_chars(text)
            texts.append(text)
            image_paths.append(image_path)
        return self._create_anns(image_paths, texts)

    def create_synth_anns(self, images_dir: str, gt_txt_path: str) -> dict[str, str]:
        lines = FileProcessor.read_txt(gt_txt_path, strip=True)
        texts, image_paths = [], []

        for line in tqdm(lines, desc="Creating annotations from synthesized data"):
            image_path, text = line.split("\t", maxsplit=1)
            image_path = os.path.join(
                images_dir, os.path.relpath(image_path, start="images")
            )

            text = text.translate(self.replaces_table)
            self.check_length(text)
            self.check_chars(text)
            texts.append(text)
            image_paths.append(image_path)
        return self._create_anns(image_paths, texts)
