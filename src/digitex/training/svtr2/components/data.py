import os
import shutil
import random

from urllib.parse import unquote

from tqdm import tqdm

from digitex.core.processors.file import FileProcessor


class AnnotationCreator:
    def __init__(
        self, charset: set[str], replaces_json_path: str, max_text_length: int
    ) -> None:
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
            raise ValueError(
                f"Length of text {text} is greater than {self.max_text_length}."
            )

    def check_chars(self, text: str) -> None:
        for char in text:
            if char not in self.charset:
                raise ValueError(f"Char {char} not in charset.")

        return None

    def __create_gt(self, image_paths: list[str], texts: list[str]) -> None:
        # Empty list to store gt
        gt = {}

        # Iterating through texts, image_paths and write to txt
        for image_path, text in zip(image_paths, texts):
            gt[image_path] = text

        return gt

    def __get_text(self, task: dict) -> str:
        # Retrieve result and label
        result = task["annotations"][0]["result"]
        text = result[0]["value"]["text"][0]

        return text

    def __get_image_path(self, task: dict, images_dir: str) -> str:
        image_path = unquote(os.path.basename(task["data"]["captioning"]))
        image_path = os.path.join(images_dir, image_path)

        return image_path

    def create_ls_gt(self, images_dir: str, data_json_path: str) -> dict[str, str]:
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
        gt = self.__create_gt(image_paths, texts)

        return gt

    def create_synth_gt(self, images_dir: str, gt_txt_path: str) -> dict[str, str]:
        lines = FileProcessor.read_txt(gt_txt_path, strip=True)

        texts = []
        image_paths = []

        for line in tqdm(lines, desc="Creating annotations from synthesized data"):
            image_path, text = line.split("\t", maxsplit=1)
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
        gt = self.__create_gt(image_paths, texts)

        return gt
