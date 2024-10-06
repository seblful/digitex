import os

from PIL import Image

import albumentations as A

import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import Mask2FormerImageProcessor, AutoImageProcessor
from transformers.image_processing_base import BatchFeature


class Mask2FormerDataset(Dataset):
    def __init__(self,
                 set_dir: str,
                 pretrained_model_name: str = None,
                 image_height: int = 384,
                 image_width: int = 384) -> None:
        # Dirs, paths with images, masks and classes
        self.set_dir = set_dir
        self.images_dir = os.path.join(set_dir, 'images')
        self.annotation_dir = os.path.join(set_dir, 'annotations')

        # List of images and annotations names
        self.images_listdir = [image for image in os.listdir(self.images_dir)]
        self.annotation_listdir = [
            annotation for annotation in os.listdir(self.annotation_dir)]

        # Assert if number of images and annotation is the same
        assert len(self.images_listdir) == len(
            self.annotation_listdir), "Number of images must be equal number of annotations."

        # Feauture extractor
        self.processor = Mask2FormerImageProcessor.from_pretrained(pretrained_model_name,
                                                                   do_resize=True,
                                                                   do_reduce_labels=False)
        self.processor.size = {"height": image_height, "width": image_width}

        # Transform and augment
        self.train_transform = A.Compose([A.HorizontalFlip(p=0.5),
                                          A.RandomBrightnessContrast(p=0.5),
                                          A.HueSaturationValue(p=0.1)])

        self.val_transform = A.Compose([A.NoOp()])

        # Remove batch dimension
        self.remove_batch_dim = ["pixel_values", "pixel_mask"]

    def __len__(self) -> int:
        return len(self.images_listdir)

    def __getitem__(self, idx: int) -> BatchFeature:
        # Open image
        image = Image.open(os.path.join(
            self.images_dir, self.images_listdir[idx]))
        img = np.array(image)

        # Open annotation
        annotation = Image.open(
            os.path.join(self.annotation_dir, self.annotation_listdir[idx]))
        semantic_and_instance_masks = np.array(annotation)[..., :2]
        instance_mask = semantic_and_instance_masks[..., 1]
        unique_semantic_id_instance_id_pairs = np.unique(
            semantic_and_instance_masks.reshape(-1, 2), axis=0)

        # Inctsnce id to semantic id
        instance_id_to_semantic_id = {
            inst_id: sem_id for sem_id, inst_id in unique_semantic_id_instance_id_pairs}

        # print(instance_id_to_semantic_id)

        # Create encoded inputs
        inputs = self.processor(images=[img],
                                segmentation_maps=[instance_mask],
                                instance_id_to_semantic_id=instance_id_to_semantic_id,
                                return_tensors="pt")

        # Remove batch dimension
        for k in self.remove_batch_dim:
            inputs[k].squeeze_()

        return inputs
