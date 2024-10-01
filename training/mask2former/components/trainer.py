import os
import math

from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import Mask2FormerModel, get_scheduler

from accelerate import Accelerator
from accelerate.utils import set_seed

from .dataset import Mask2FormerDataset


class Mask2FormerTrainer:
    def __init__(self,
                 dataset_dir: str,
                 pretrained_model_name: str = None,
                 batch_size: int = 4,
                 num_workers: int = 1,
                 learning_rate: float = 0.001,
                 lr_scheduler_type: str = "constant",
                 train_epochs: int = 10,
                 train_steps: int = None,
                 gradient_accumulation_steps: int = 1,
                 warmup_steps: int = 0,
                 checkpoint_steps: int = 10,
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
        self.seed = seed

        # Model
        self.pretrained_model_name = pretrained_model_name
        self.model = Mask2FormerModel.from_pretrained(pretrained_model_name,
                                                      label2id=self.label2id,
                                                      id2label=self.id2label,
                                                      ignore_mismatched_sizes=True)

        # Training parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Data
        self.__setup_data()

        # Epochs and steps
        self.train_epochs = train_epochs

        self.train_steps = train_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.checkpoint_steps = checkpoint_steps
        self.__setup_steps()

        # LR, Optimizer
        self.accelerator = Accelerator(mixed_precision=None,  # TODO add mixed precision
                                       gradient_accumulation_steps=gradient_accumulation_steps)
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.__setup_optimizers()

        # Setup accelerator
        self.__setup_accelerator()

    def __setup_data(self) -> None:
        train_dataset = Mask2FormerDataset(set_dir=self.train_dir,
                                           pretrained_model_name=self.pretrained_model_name)
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers)

        val_dataset = Mask2FormerDataset(set_dir=self.val_dir,
                                         pretrained_model_name=self.pretrained_model_name)

        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         num_workers=self.num_workers)

        test_dataset = Mask2FormerDataset(set_dir=self.test_dir,
                                          pretrained_model_name=self.pretrained_model_name)

        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=self.num_workers)

    def __setup_steps(self) -> None:
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.gradient_accumulation_steps)

        self.overrode_train_steps = False
        if self.train_steps is None:
            self.train_steps = self.train_epochs * num_update_steps_per_epoch
            self.overrode_train_steps = True

    def __setup_optimizers(self) -> None:
        # Optimizer
        self.optimizer = AdamW(list(self.model.parameters()),
                               lr=self.learning_rate)

        # LR scheduler
        num_training_steps = self.train_steps if self.overrode_train_steps else self.train_steps * \
            self.accelerator.num_processes,
        self.lr_scheduler = get_scheduler(name=self.lr_scheduler_type,
                                          optimizer=self.optimizer,
                                          num_warmup_steps=self.warmup_steps * self.accelerator.num_processes,
                                          num_training_steps=num_training_steps)

    def __setup_accelerator(self) -> None:
        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.lr_scheduler)

        # Recalculate total training steps
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.gradient_accumulation_steps)

        if self.overrode_train_steps:
            self.train_steps = self.train_epochs * num_update_steps_per_epoch

        # Recalculate number of training epochs
        self.train_epochs = math.ceil(
            self.train_steps / num_update_steps_per_epoch)

        # Set seed
        set_seed(self.seed, device_specific=True)

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
