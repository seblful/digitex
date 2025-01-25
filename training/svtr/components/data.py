import os
import shutil
import random

from urllib.parse import unquote

from tqdm import tqdm

from modules.processors import FileProcessor


class DatasetCreator():
    def __init__(self,
                 raw_dir: str,
                 dataset_dir: str,
                 train_split: float = 0.8,
                 max_text_length=31) -> None:

        # Input paths
        self.raw_dir = raw_dir
        self.chars_txt_path = os.path.join(raw_dir, "chars.txt")
        self.replaces_json_path = os.path.join(
            raw_dir, "replaces.json")

        # Output paths
        self.dataset_dir = dataset_dir
        self.charset_txt_path = os.path.join(dataset_dir, "charset.txt")

        # Data split
        self.__setup_splits(train_split)

        # Sources
        self.sources = ["ls", "synth"]

        # Annotation creator
        self.annotation_creator = AnnotationCreator(charset=self.charset,
                                                    replaces_json_path=self.replaces_json_path,
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

    @property
    def charset(self) -> set[str]:
        if self.__charset is None:
            charset = FileProcessor.read_txt(self.chars_txt_path)[0].strip()
            self.__charset = set(charset)

        return self.__charset

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
        dir_name = os.path.basename(set_dir)

        # Create list to store gt lines for txt gt
        gt_lines = []

        # Sort dict and iterate through it
        gt = dict(sorted(gt.items(), key=self.sort_gt_key))

        for image_path, text in tqdm(gt.items(), desc=f"Partitioning {dir_name} data"):
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

    def __create_charset(self) -> None:
        chars = sorted(self.charset)

        # Create lines with char and new line
        charlines = []
        for char in chars:
            charline = char + "\n"
            charlines.append(charline)

        # Write lines to txt
        FileProcessor.write_txt(self.charset_txt_path, charlines)

        return None

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
            gt_dict.keys())[num_train:]}

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
        if source == "ls":
            data_json_path = os.path.join(self.raw_dir, 'data.json')
            self.annotation_creator.create_ls_gt(
                data_json_path=data_json_path)
        else:
            gt_txt_path = os.path.join(self.raw_dir, "gt.txt")
            self.annotation_creator.create_synth_gt(
                gt_txt_path=gt_txt_path)

        # Create dataset
        self.__partitionate_data()

        # Create charser
        self.__create_charset()


class AnnotationCreator:
    def __init__(self,
                 charset: set[str],
                 replaces_json_path: str,
                 max_text_length: int) -> None:
        self.charset = charset

        # Paths
        self.replaces_json_path = replaces_json_path

        self.max_text_length = max_text_length

        self.__replaces_table = None

    @property
    def replaces_table(self) -> dict:
        if self.__replaces_table is None:
            replaces_dict = FileProcessor.read_json(self.replaces_json_path)
            self.__replaces_table = str.maketrans(replaces_dict)

        return self.__replaces_table

    def check_length(self, text: str) -> None:
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

        return gt

    def __get_text(self, task: dict) -> str:
        # Retrieve result and label
        result = task['annotations'][0]['result']
        text = result[0]['value']['text'][0]

        return text

    def __get_image_path(self, task: dict, images_dir: str) -> str:
        image_path = unquote(os.path.basename(task["data"]["captioning"]))
        image_path = os.path.join(images_dir, image_path)

        return image_path

    def create_ls_gt(self,
                     images_dir: str,
                     data_json_path: str) -> dict[str, str]:
        # Read jsons
        json_dict = FileProcessor.read_json(data_json_path)

        texts = []
        image_paths = []

        # Iterate through task
        for task in tqdm(json_dict, desc="Creating annotations from Label Studio"):
            image_path = self.__get_image_path(task, images_dir)
            text = self.__get_text(task)

            # Replace chars that not in the charset
            text = text.translate(self.replaces_table)

            # Check text length, chars in charset
            self.check_length(text)
            self.check_chars(text)

            texts.append(text)
            image_paths.append(image_path)

        # Create GT labels
        gt = self.create_gt(image_paths, texts)

        return gt

    def create_synth_gt(self,
                        images_dir: str,
                        gt_txt_path: str) -> dict[str, str]:
        lines = FileProcessor.read_txt(gt_txt_path)

        texts = []
        image_paths = []

        for line in tqdm(lines, desc="Creating annotations from synthesized data"):
            line = line.strip()
            image_path, text = line.split(maxsplit=1)
            image_path = os.path.relpath(image_path, start="images")
            image_path = os.path.join(images_dir, image_path)

            # Replace chars that not in the charset
            text = text.translate(self.replaces_table)

            # Check text length, chars in charset
            self.check_length(text)
            self.check_chars(text)

            texts.append(text)
            image_paths.append(image_path)

        # Create GT labels
        gt = self.create_gt(image_paths, texts)

        return gt
