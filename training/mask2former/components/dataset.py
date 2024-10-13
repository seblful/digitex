import os
from PIL import Image

import albumentations as A

import numpy as np

from torch.utils.data import Dataset

from transformers import Mask2FormerImageProcessor
from transformers.image_processing_base import BatchFeature


class Mask2FormerDataset(Dataset):
    def __init__(self,
                 set_dir: str,
                 image_processor: Mask2FormerImageProcessor) -> None:

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

        # Image processor
        self.image_processor = image_processor

        # Transform and augment
        self.train_transform = A.Compose([A.HorizontalFlip(p=0.5),
                                          A.RandomBrightnessContrast(p=0.5),
                                          A.HueSaturationValue(p=0.1)])

        self.val_transform = A.Compose([A.NoOp()])

        # Remove batch dimension
        self.remove_batch_dim = ["pixel_values", "pixel_mask"]
        self.remove_nesting = ["mask_labels", "class_labels"]

    def __len__(self) -> int:
        return len(self.images_listdir)

    def __getitem__(self, idx: int) -> BatchFeature:
        image = Image.open(os.path.join(
            self.images_dir, self.images_listdir[idx]))
        annotation = Image.open(os.path.join(
            self.annotation_dir, self.annotation_listdir[idx]))

        # Convert to numpy array
        image_array = np.array(image)
        annotation_array = np.array(annotation)

        # Extract instance and semantic segmentation
        instance_seg = annotation_array[..., 1]
        semantic_seg = annotation_array[..., 0]

        # Get unique instance ids
        instance_ids = np.unique(instance_seg)

        # Create instance_id_to_semantic_id mapping
        instance_id_to_semantic_id = {
            inst_id: semantic_seg[instance_seg == inst_id][0] for inst_id in instance_ids}

        # # Apply transformations if any
        # if self.transform:
        #     transformed = self.transform(
        #         image=np.array(image), mask=instance_seg)
        #     image = transformed['image']
        #     instance_seg = transformed['mask']

        # Prepare inputs for the model
        inputs = self.image_processor(images=image_array,
                                      segmentation_maps=instance_seg,
                                      instance_id_to_semantic_id=instance_id_to_semantic_id,
                                      return_tensors="pt")

        # Remove batch dimension
        for k in self.remove_batch_dim:
            inputs[k].squeeze_()

        # Remove list nesting
        for k in self.remove_nesting:
            inputs[k] = inputs[k][0]

        return inputs
