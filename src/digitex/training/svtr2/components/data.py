import os
import shutil
import random
from urllib.parse import unquote
from tqdm import tqdm
from digitex.core.processors.file import FileProcessor


class DatasetCreator:
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

    def _setup_dataset_dirs(self) -> None:
        os.mkdir(self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.dataset_dirs = (self.train_dir, self.val_dir)
        for dir in self.dataset_dirs:
            os.mkdir(dir)
            images_dir = os.path.join(dir, "images")
            os.mkdir(images_dir)

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
    def sort_gt_key(key) -> tuple:
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

    def _copy_data(
        self, gt: dict[str, str], set_dir: str, images_per_folder: int = 10000
    ) -> None:
        dir_name = os.path.basename(set_dir)
        images_dir = os.path.join(set_dir, "images")
        gt_lines = []
        gt = dict(sorted(gt.items(), key=self.sort_gt_key))
        subfolder_idx = img_in_subfolder = 0
        for idx, (image_path, text) in enumerate(
            tqdm(gt.items(), desc=f"Partitioning {dir_name} data")
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
            shutil.copyfile(src_image_path, os.path.join(set_dir, dst_image_path))
            gt_lines.append(f"{dst_image_path}\t{text}")
            img_in_subfolder += 1
        gt_txt_path = os.path.join(set_dir, "gt.txt")
        FileProcessor.write_txt(txt_path=gt_txt_path, lines=gt_lines, newline=True)

    def _partitionate_data(
        self, gt_dict: dict[str, str], aug_gt_dict: dict[str, str] = None
    ) -> None:
        gt_dict = self.shuffle_dict(gt_dict)
        num_train = int(len(gt_dict) * self.train_split)
        train_keys = list(gt_dict.keys())[:num_train]
        val_keys = list(gt_dict.keys())[num_train:]
        train_gt_dict = {key: gt_dict[key] for key in train_keys}
        val_gt_dict = {key: gt_dict[key] for key in val_keys}
        if aug_gt_dict:
            train_gt_dict.update(aug_gt_dict)
            train_gt_dict = self.shuffle_dict(train_gt_dict)
        for gt, set_dir in zip((train_gt_dict, val_gt_dict), self.dataset_dirs):
            self._copy_data(gt=gt, set_dir=set_dir)

    def create_dataset(self, source: str, use_aug=False) -> None:
        assert source in self.sources, (
            f"Source of raw images must be one of {self.sources}."
        )
        self._setup_dataset_dirs()
        if source == "ls":
            images_dir = "images"
            data_json_path = os.path.join(self.raw_dir, "data.json")
            gt_dict = self.annotation_creator.create_ls_gt(
                images_dir=images_dir, data_json_path=data_json_path
            )
        else:
            images_dir = "images"
            gt_txt_path = os.path.join(self.raw_dir, "gt.txt")
            gt_dict = self.annotation_creator.create_synth_gt(
                images_dir=images_dir, gt_txt_path=gt_txt_path
            )
        aug_gt_dict = None
        if use_aug:
            images_dir = "aug-images"
            gt_txt_path = os.path.join(self.raw_dir, "aug_gt.txt")
            aug_gt_dict = self.annotation_creator.create_synth_gt(
                images_dir=images_dir, gt_txt_path=gt_txt_path
            )
        self._partitionate_data(gt_dict=gt_dict, aug_gt_dict=aug_gt_dict)
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

    def _create_gt(self, image_paths: list[str], texts: list[str]) -> dict[str, str]:
        return dict(zip(image_paths, texts))

    def _get_text(self, task: dict) -> str:
        return task["annotations"][0]["result"][0]["value"]["text"][0]

    def _get_image_path(self, task: dict, images_dir: str) -> str:
        image_path = unquote(os.path.basename(task["data"]["captioning"]))
        return os.path.join(images_dir, image_path)

    def create_ls_gt(self, images_dir: str, data_json_path: str) -> dict[str, str]:
        json_dict = FileProcessor.read_json(data_json_path)
        texts, image_paths = [], []
        for task in tqdm(json_dict, desc="Creating annotations from Label Studio"):
            image_path = self._get_image_path(task, images_dir)
            text = self._get_text(task).translate(self.replaces_table)
            self.check_length(text)
            self.check_chars(text)
            texts.append(text)
            image_paths.append(image_path)
        return self._create_gt(image_paths, texts)

    def create_synth_gt(self, images_dir: str, gt_txt_path: str) -> dict[str, str]:
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
        return self._create_gt(image_paths, texts)
