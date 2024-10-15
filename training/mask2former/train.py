import os
import argparse

from components.trainer import Mask2FormerTrainer

HOME = os.getcwd()
DATASET_DIR = os.path.join(HOME, "data", "dataset")
RUNS_DIR = os.path.join(HOME, "runs")

PRETRAINED_MODEL = "facebook/mask2former-swin-tiny-coco-panoptic"


def main() -> None:
    trainer = Mask2FormerTrainer(dataset_dir=DATASET_DIR,
                                 runs_dir=RUNS_DIR,
                                 pretrained_model_name=PRETRAINED_MODEL,
                                 image_height=640,
                                 image_width=640,
                                 batch_size=2,
                                 learning_rate=0.0001,
                                 lr_scheduler_type="cosine",
                                 mixed_precision="no",
                                 train_epochs=100,
                                 checkpoint_steps=500,
                                 seed=2)
    trainer.train()


if __name__ == "__main__":
    main()
