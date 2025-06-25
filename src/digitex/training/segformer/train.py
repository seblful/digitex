import os
import yaml

from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from digitex.settings import settings

from digitex.training.segformer.components.dataset import SegFormerDataset
from digitex.training.segformer.components.trainer import SegFormerTrainer

HOME = os.getcwd()
SEGFORMER_DIR = os.path.join(HOME, "src", "digitex", "training", "segformer")
DATASET_DIR = os.path.join(SEGFORMER_DIR, "data", "dataset")
TRAIN_DATASET_DIR = os.path.join(DATASET_DIR, "train")
VAL_DATASET_DIR = os.path.join(DATASET_DIR, "val")

CONFIG_PATH = os.path.join(SEGFORMER_DIR, "config", "config.yml")


def train() -> None:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = settings.DEVICE
    print(f"Using device: {device}.")

    # Create datasets
    train_dataset = SegFormerDataset(
        dataset_dir=TRAIN_DATASET_DIR,
        model_name=config["model"]["model_name"],
        image_size=tuple(config["dataset"]["image_size"]),
    )

    val_dataset = SegFormerDataset(
        dataset_dir=VAL_DATASET_DIR,
        model_name=config["model"]["model_name"],
        image_size=tuple(config["dataset"]["image_size"]),
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize SegFormer model
    model = SegformerForSemanticSegmentation.from_pretrained(
        config["model"]["model_name"],
        num_labels=config["dataset"]["num_classes"],
        id2label=config["model"]["id2label"],
        label2id=config["model"]["label2id"],
        ignore_mismatched_sizes=True,
    )

    print(f"Loaded SegFormer model: {config['model']['model_name']}")
    print(f"Number of classes: {config['dataset']['num_classes']}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["trainer"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["trainer"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=True if device == "cuda" else False,
    )

    # Initialize trainer
    trainer = SegFormerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=config["dataset"]["num_classes"],
        lr=config["trainer"]["lr"],
        weight_decay=config["trainer"]["weight_decay"],
        log_dir=config["trainer"]["log_dir"],
        checkpoint_dir=config["trainer"]["checkpoint_dir"],
        checkpoint_path=config["trainer"]["checkpoint_path"],
        use_tensorboard=config["trainer"]["use_tensorboard"],
        use_custom_loss=config["trainer"].get("use_custom_loss", True),
        loss_type=config["trainer"].get("loss_type", "combined"),
    )

    # Start training
    trainer.train(
        num_epochs=config["trainer"]["num_epochs"],
        save_every=config["trainer"]["save_every"],
        early_stopping=config["trainer"]["early_stopping"],
    )


def main() -> None:
    train()


if __name__ == "__main__":
    main()
