import logging
from pathlib import Path

import typer
from components.trainer import Trainer
from modules.config import get_settings

app = typer.Typer(help="YOLO model training for document segmentation")

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.app.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_training_defaults():
    """Get default training parameters from settings."""
    settings = get_settings()
    return settings.training


@app.command()
def main(
    data_subdir: str = typer.Option(
        None,
        "--data-subdir",
        help="Type of task type (e.g., 'page', 'question', 'part')",
    ),
    model_type: str = typer.Option(
        None,
        "--model-type",
        help="Type of YOLO model ('seg')",
    ),
    model_size: str = typer.Option(
        None,
        "--model-size",
        help="Size of YOLO model ('n', 's', 'm', 'l', 'x')",
    ),
    pretrained_model_path: str = typer.Option(
        None,
        "--pretrained-model-path",
        help="Path to a previously trained model",
    ),
    num_epochs: int = typer.Option(
        None,
        "--num-epochs",
        help="Number of training epochs",
    ),
    image_size: int = typer.Option(
        None,
        "--image-size",
        help="Input image size for training",
    ),
    batch_size: int = typer.Option(
        None,
        "--batch-size",
        help="Batch size for training",
    ),
    overlap_mask: bool = typer.Option(
        None,
        "--overlap-mask",
        help="Whether segmentation masks should overlap",
    ),
    patience: int = typer.Option(
        None,
        "--patience",
        help="Early stopping patience in epochs",
    ),
    seed: int = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducibility",
    ),
) -> None:
    """Train a YOLO model for document segmentation."""
    setup_logging()

    settings = get_settings()
    train_defaults = get_training_defaults()

    data_subdir = data_subdir or train_defaults.data_subdir
    model_type = model_type or train_defaults.model_type
    model_size = model_size or train_defaults.model_size
    pretrained_model_path = pretrained_model_path or train_defaults.pretrained_model_path
    num_epochs = num_epochs or train_defaults.num_epochs
    image_size = image_size or train_defaults.image_size
    batch_size = batch_size or train_defaults.batch_size
    overlap_mask = overlap_mask if overlap_mask is not None else train_defaults.overlap_mask
    patience = patience or train_defaults.patience
    seed = seed or train_defaults.seed

    data_dir = settings.paths.data_dir / data_subdir
    dataset_dir = data_dir / "dataset"

    logger.info(f"Starting YOLO training")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Dataset directory: {dataset_dir}")

    trainer = Trainer(
        dataset_dir=dataset_dir,
        model_type=model_type,
        model_size=model_size,
        pretrained_model_path=pretrained_model_path,
        num_epochs=num_epochs,
        image_size=image_size,
        batch_size=batch_size,
        overlap_mask=overlap_mask,
        patience=patience,
        seed=seed,
    )

    try:
        trainer.train()
        trainer.validate()
        logger.info("Training and validation completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
