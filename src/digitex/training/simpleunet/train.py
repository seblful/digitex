import os
import yaml

from torch.utils.data import DataLoader

from digitex.settings import settings

from digitex.training.simpleunet.components.dataset import MaskDataset
from digitex.training.simpleunet.components.model import SimpleUNet
from digitex.training.simpleunet.components.trainer import MaskSegmentationTrainer

HOME = os.getcwd()
SIMPLEUNET_DIR = os.path.join(HOME, "src", "digitex", "training", "simpleunet")
DATASET_DIR = os.path.join(SIMPLEUNET_DIR, "data", "dataset")
TRAIN_DATASET_DIR = os.path.join(DATASET_DIR, "train")
VAL_DATASET_DIR = os.path.join(DATASET_DIR, "val")

CONFIG_PATH = os.path.join(
    HOME, "src", "digitex", "training", "simpleunet", "config", "config.yml"
)


def train() -> None:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = settings.DEVICE

    train_dataset = MaskDataset(
        dataset_dir=TRAIN_DATASET_DIR,
        image_size=tuple(config["dataset"]["image_size"]),
        mask_size=tuple(config["dataset"]["mask_size"]),
    )

    val_dataset = MaskDataset(
        dataset_dir=VAL_DATASET_DIR,
        image_size=tuple(config["dataset"]["image_size"]),
        mask_size=tuple(config["dataset"]["mask_size"]),
    )

    model = SimpleUNet(
        in_channels=config["model"]["in_channels"],
        num_cls=config["model"]["num_cls"],
        ks=config["model"]["ks"],
        dilation=config["model"]["dilation"],
        stage_channels=config["model"]["stage_channels"],
        num_blocks=config["model"]["num_blocks"],
        short_rate=config["model"]["short_rate"],
        adw=config["model"]["adw"],
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

    trainer = MaskSegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=config["trainer"]["lr"],
        weight_decay=config["trainer"]["weight_decay"],
        log_dir=config["trainer"]["log_dir"],
        checkpoint_dir=config["trainer"]["checkpoint_dir"],
        checkpoint_path=config["trainer"]["checkpoint_path"],
        use_tensorboard=config["trainer"]["use_tensorboard"],
        iou_threshold=config["trainer"]["iou_threshold"],
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
