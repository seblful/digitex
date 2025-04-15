import os

import torch
from ultralytics import YOLO


class Trainer():
    def __init__(self,
                 dataset_dir: str,
                 model_type: str,
                 model_size: str,
                 num_epochs: int,
                 image_size: int,
                 batch_size: int,
                 pretrained_model_path: str = None,
                 overlap_mask: bool = False,
                 patience: int = 50,
                 seed: int = 42) -> None:

        # Data
        self.dataset_dir = dataset_dir
        self.__data = None

        # Parameters
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.batch_size = batch_size
        self.overlap_mask = overlap_mask
        self.patience = patience
        self.seed = seed

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Current device is {self.device}.")
        self.device_count = torch.cuda.device_count()
        self.device_idxs = [i for i in range(self.device_count)]

        # Model
        self.pretrained_model_path = pretrained_model_path
        self.model_yaml = f"yolo11{model_size}-{model_type}.yaml"
        self.model_pt = f"yolo11{model_size}-{model_type}.pt"

        self.__model = None
        self.is_trained = False

    @property
    def model(self) -> YOLO:
        if self.__model == None:
            if self.pretrained_model_path is None:
                model = YOLO(self.model_yaml).load(self.model_pt)
            else:
                model = YOLO(self.pretrained_model_path)

            self.__model = model

        return self.__model

    @property
    def data(self) -> str:
        if self.__data is None:
            self.__data = os.path.join(self.dataset_dir, "data.yaml")

        return self.__data

    def train(self) -> None:
        self.model.train(data=self.data,
                         epochs=self.num_epochs,
                         imgsz=self.image_size,
                         batch=self.batch_size,
                         overlap_mask=self.overlap_mask,
                         patience=self.patience,
                         device=self.device_idxs,
                         seed=self.seed)

        self.is_trained = True

        return None

    def validate(self) -> None:
        if self.is_trained is False:
            raise ValueError("Model must be trained before validating.")

        self.model.val(data=self.data,
                       imgsz=self.image_size,
                       split='test')

        return
