import os

import torch
import ultralytics


class Trainer():
    def __init__(self,
                 dataset_dir,
                 num_epochs,
                 image_size,
                 batch_size,
                 seed,
                 model_type='n'):

        self.dataset_dir = dataset_dir
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.batch_size = batch_size
        self.model_type = model_type

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.seed = seed

        self.model_yaml = f"yolov8{model_type}-seg.yaml"
        self.model_type = f"yolov8{model_type}-seg.pt"
        self.__data = None

        self.__model = None
        self.is_trained = False

    @property
    def model(self):
        if self.__model == None:
            # Build a new model from scratch
            model = ultralytics.YOLO(self.model_yaml)
            # Load a pretrained model
            model = ultralytics.YOLO(self.model_type)

            # # Load from my pretrained model
            # model = ultralytics.YOLO('best3.pt')

            self.__model = model

        return self.__model

    @property
    def data(self):
        if self.__data is None:
            self.__data = os.path.join(self.dataset_dir, "data.yaml")

        return self.__data

    def train(self):
        '''
        Training model
        '''

        result = self.model.train(data=self.data, epochs=self.num_epochs, imgsz=self.image_size,
                                  batch=self.batch_size, seed=self.seed)  # close_mosaic=0, workers=1
        self.is_trained = True

        return result

    def validate(self):
        '''
        Validating model on test dataset
        '''

        if self.is_trained == True:
            metrics = self.model.val(
                data=self.data, imgsz=self.image_size, split='test')

            return metrics

        else:
            return f"Model is not trained yet."
