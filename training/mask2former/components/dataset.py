import os

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from transformers import Mask2FormerImageProcessor
from transformers.image_processing_base import BatchFeature


class Mask2FormerDataset(Dataset):
    def __init__(self,
                 set_dir: str,
                 image_height: int,
                 image_width: int) -> None:
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
            self.masks_listdir), "Number of images must be equal number of annotations."

        # Feauture extractor
        self.processor = Mask2FormerImageProcessor(do_reduce_labels=False)
        self.processor.size = {"height": image_height, "width": image_width}

    def __len__(self) -> int:
        return len(self.images_listdir)

    def __getitem__(self, idx: int) -> BatchFeature:
        # Open image
        image = Image.open(os.path.join(
            self.images_dir, self.images_listdir[idx]))

        # Open annotation
        annotation = Image.open(
            os.path.join(self.annotation_dir, self.annotation_listdir[idx]))

        # Create encoded inputs
        inputs = self.processor(
            image, annotation, return_tensors="pt")

        # for k, v in inputs.items():
        #     inputs[k].squeeze_()  # remove batch dimension

        return inputs
