import os
import argparse

from components.trainer import Mask2FormerTrainer

HOME = os.getcwd()
DATASET_DIR = os.path.join(HOME, "data", "dataset")
RUNS_DIR = os.path.join(HOME, "runs")

PRETRAINED_MODEL = "facebook/mask2former-swin-base-coco-panoptic"


def main() -> None:
    trainer = Mask2FormerTrainer(dataset_dir=DATASET_DIR,
                                 runs_dir=RUNS_DIR,
                                 pretrained_model_name=PRETRAINED_MODEL)
    trainer.train()


if __name__ == "__main__":
    main()
