import os
import json

from components.trainer import OneFormerTrainer

HOME = os.getcwd()
DATASET_DIR = os.path.join(HOME, "data", "dataset")
RUNS_DIR = os.path.join(HOME, "runs")

PREPROCESSOR_CONFIG_PATH = "preprocessor_config.json"
CLASS_INFO_FILE_PATH = "class_info_file.json"

PRETRAINED_MODEL = "shi-labs/oneformer_ade20k_swin_tiny"


def main() -> None:

    trainer = OneFormerTrainer(dataset_dir=DATASET_DIR,
                               runs_dir=RUNS_DIR,
                               preprocessor_config_path=PREPROCESSOR_CONFIG_PATH,
                               class_info_file_path=CLASS_INFO_FILE_PATH,
                               pretrained_model_name=PRETRAINED_MODEL,
                               longest_edge=2048,
                               shortest_edge=512,
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
