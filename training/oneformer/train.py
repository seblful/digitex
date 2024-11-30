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
                               width=1024,
                               height=256,
                               batch_size=1,
                               learning_rate=5e-5,
                               lr_scheduler_type="cosine",
                               mixed_precision="no",
                               train_epochs=3,
                               seed=2)
    trainer.train()


if __name__ == "__main__":
    main()
