import logging
from pathlib import Path

import typer
from components.trainer import Trainer

app = typer.Typer(help="YOLO model training for document segmentation")

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def main(
    data_subdir: str = typer.Option(
        "page",
        "--data-subdir",
        help="Type of task type (e.g., 'page', 'question', 'part')",
    ),
    model_type: str = typer.Option(
        "seg",
        "--model-type",
        help="Type of YOLO model ('seg', 'obb', 'pose')",
    ),
    model_size: str = typer.Option(
        "m",
        "--model-size",
        help="Size of YOLO model ('n', 's', 'm', 'l', 'x')",
    ),
    pretrained_model_path: str = typer.Option(
        None,
        "--pretrained-model-path",
        help="Path to a previously trained model",
    ),
    num_epochs: int = typer.Option(
        100,
        "--num-epochs",
        help="Number of training epochs",
    ),
    image_size: int = typer.Option(
        640,
        "--image-size",
        help="Input image size for training",
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        help="Batch size for training",
    ),
    overlap_mask: bool = typer.Option(
        False,
        "--overlap-mask",
        help="Whether segmentation masks should overlap",
    ),
    patience: int = typer.Option(
        50,
        "--patience",
        help="Early stopping patience in epochs",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
) -> None:
    """Train a YOLO model for document segmentation."""
    setup_logging()

    home = Path.cwd()
    data_dir = home / "data" / data_subdir
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
