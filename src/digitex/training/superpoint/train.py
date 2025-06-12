import os
import yaml

import torch
from torch.utils.data import DataLoader

from digitex.training.superpoint.components.dataset import HeatmapKeypointDataset
from digitex.training.superpoint.components.model import HeatmapKeypointModel
from digitex.training.superpoint.components.trainer import HeatmapKeypointTrainer
from digitex.training.superpoint.components.utils import visualize_sample

HOME = os.getcwd()
SUPERPOINT_DIR = os.path.join(HOME, "src", "digitex", "training", "superpoint")
DATASET_DIR = os.path.join(SUPERPOINT_DIR, "data", "dataset")
TRAIN_DATASET_DIR = os.path.join(DATASET_DIR, "train")
VAL_DATASET_DIR = os.path.join(DATASET_DIR, "val")

# Load config from YAML
CONFIG_PATH = os.path.join(
    os.getcwd(), "src", "digitex", "training", "superpoint", "config", "config.yml"
)


def train() -> None:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = HeatmapKeypointDataset(
        dataset_dir=TRAIN_DATASET_DIR,
        image_size=tuple(config["dataset"]["image_size"]),
        heatmap_size=tuple(config["dataset"]["heatmap_size"]),
        max_keypoints=config["dataset"]["max_keypoints"],
        load_heatmaps=config.get("load_heatmaps", False),
        heatmap_dir=config.get("heatmap_dir", None),
        sigma=config["dataset"]["heatmap_sigma"],
    )

    val_dataset = HeatmapKeypointDataset(
        dataset_dir=VAL_DATASET_DIR,
        image_size=tuple(config["dataset"]["image_size"]),
        heatmap_size=tuple(config["dataset"]["heatmap_size"]),
        max_keypoints=config["dataset"]["max_keypoints"],
        load_heatmaps=config.get("load_heatmaps", False),
        heatmap_dir=config.get("heatmap_dir", None),
        sigma=config["dataset"]["heatmap_sigma"],
    )

    model = HeatmapKeypointModel(
        max_keypoints=config["dataset"]["max_keypoints"],
        backbone_out_channels=config["model"]["backbone_out_channels"],
        backbone_stride=config["model"]["backbone_stride"],
        deconv_channels=config["model"]["deconv_channels"],
        output_stride=config["model"]["output_stride"],
    ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["trainer"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["trainer"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )

    trainer = HeatmapKeypointTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_keypoints=config["dataset"]["max_keypoints"],
        device=device,
        lr=config["trainer"]["lr"],
        weight_decay=config["trainer"]["weight_decay"],
        log_dir=config["trainer"]["log_dir"],
        checkpoint_dir=config["trainer"]["checkpoint_dir"],
    )
    trainer.train(
        num_epochs=config["trainer"]["num_epochs"],
        save_every=config["trainer"]["save_every"],
        early_stopping=config["trainer"]["early_stopping"],
    )


def main() -> None:
    train()


if __name__ == "__main__":
    main()
