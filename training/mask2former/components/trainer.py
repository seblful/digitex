import os

from transformers import AutoImageProcessor, Mask2FormerModel

from accelerate import Accelerator
from accelerate.utils import set_seed


class Mask2FormerTrainer:
    def __init__(self,
                 dataset_dir: str,
                 pretrained_model_name: str = None,
                 seed: int = 42) -> None:
        # Paths
        self.dataset_dir = dataset_dir

        self.train_dir = os.path.join(dataset_dir, "train")
        self.val_dir = os.path.join(dataset_dir, "val")
        self.test_dir = os.path.join(dataset_dir, "test")

        self.classes_path = os.path.join(dataset_dir, 'classes.txt')

        # ID and label
        self.__id2label = None
        self.__label2id = None

        # Seed
        self.seed = set_seed(seed, device_specific=True)

        # Accelerator
        self.accelerator = Accelerator(mixed_precision=None,  # TODO add mixed precision
                                       gradient_accumulation_steps=1)  # TODO add gradient accumulation

        # Model
        self.model = Mask2FormerModel.from_pretrained(pretrained_model_name,
                                                      label2id=self.label2id,
                                                      id2label=self.id2label,
                                                      ignore_mismatched_sizes=True)

    @property
    def id2label(self) -> dict[int, str]:
        if self.__id2label is None:
            self.__id2label = self.__create_id2label()

        return self.__id2label

    @property
    def label2id(self) -> dict[str, int]:
        if self.__label2id is None:
            self.__label2id = {v: k for k, v in self.id2label.items()}

        return self.__label2id

    def __create_id2label(self) -> dict[int, str]:
        with open(self.classes_path, 'r') as file:
            # Set the names of the classes
            classes = [i.split('\n')[0] for i in file.readlines()]
            id2label = {k: v for k, v in enumerate(classes, start=1)}

        # Set background as 0
        id2label[0] = "background"

        return id2label
