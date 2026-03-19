import logging
from pathlib import Path

import typer
from digitex.config import get_settings
from digitex.core.page_creator import PageDataCreator
from digitex.ml.yolo import Trainer
from digitex.ml.yolo.augmenter import PolygonAugmenter
from digitex.ml.yolo.dataset import DatasetCreator
from digitex.ml.yolo.visualizer import PolygonVisualizer
from digitex.utils import create_pdf_from_images

app = typer.Typer(help="YOLO model training for document segmentation")
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_training_defaults():
    """Get default training parameters from settings."""
    settings = get_settings()
    return settings.training


def get_paths(data_subdir: str = "page"):
    """Get common paths for training commands."""
    settings = get_settings()
    home = Path.cwd()
    training_dir = settings.paths.training_dir or home / "training"
    data_dir = training_dir / "data" / data_subdir
    return {
        "home": home,
        "training_dir": training_dir,
        "data_dir": data_dir,
        "raw_dir": data_dir / "raw-data",
        "dataset_dir": data_dir / "dataset",
        "check_images_dir": data_dir / "check-images",
    }


@app.command()
def create_data(
    data_subdir: str = typer.Option("page", "--data-subdir", help="Type of task type"),
    train_split: float = typer.Option(0.8, "--train-split", help="Split of train set"),
    anns_type: str = typer.Option("polygon", "--anns-type", help="Annotation type"),
    augment: bool = typer.Option(False, "--augment", help="Whether to augment train data"),
    aug_images: int = typer.Option(100, "--aug-images", help="Augmented images to create"),
    visualize: bool = typer.Option(False, "--visualize", help="Whether to visualize data"),
    vis_images: int = typer.Option(50, "--vis-images", help="Images to visualize"),
) -> None:
    """Create training dataset from raw data."""
    paths = get_paths(data_subdir)

    dataset_creator = DatasetCreator(
        raw_dir=paths["raw_dir"],
        dataset_dir=paths["dataset_dir"],
        train_split=train_split,
    )
    dataset_creator.create(anns_type=anns_type)

    if augment:
        augmenter = PolygonAugmenter(
            raw_dir=str(paths["raw_dir"]), dataset_dir=str(paths["dataset_dir"])
        )
        augmenter.augment(num_images=aug_images)

    if visualize:
        visualizer = PolygonVisualizer(
            dataset_dir=str(paths["dataset_dir"]),
            check_images_dir=str(paths["check_images_dir"]),
        )
        visualizer.visualize(num_images=vis_images)


@app.command()
def prepare_train_data(
    data_subdir: str = typer.Option("page", "--data-subdir", help="Type of task type"),
    num_images: int = typer.Option(100, "--num-images", help="Number of images to create"),
) -> None:
    """Prepare training data by creating PDFs and extracting pages."""
    settings = get_settings()
    home = Path.cwd()
    training_dir = settings.paths.training_dir or home / "training"
    books_dir = home / "books"

    images_dir = training_dir / "data" / data_subdir / "books"
    page_train_dir = training_dir / "data" / data_subdir / "images"

    page_creator = PageDataCreator()

    for image_dir in images_dir.iterdir():
        create_pdf_from_images(image_dir=image_dir, raw_dir=books_dir)

    page_creator.create(
        pdf_dir=books_dir,
        output_dir=page_train_dir,
        num_images=num_images,
    )


@app.command()
def train(
    data_subdir: str | None = typer.Option(None, "--data-subdir", help="Task type"),
    model_type: str | None = typer.Option(None, "--model-type", help="YOLO model type"),
    model_size: str | None = typer.Option(None, "--model-size", help="YOLO model size"),
    pretrained_model_path: str | None = typer.Option(None, "--pretrained-model-path", help="Model path"),
    num_epochs: int | None = typer.Option(None, "--num-epochs", help="Training epochs"),
    image_size: int | None = typer.Option(None, "--image-size", help="Input image size"),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Batch size"),
    overlap_mask: bool | None = typer.Option(None, "--overlap-mask", help="Mask overlap"),
    patience: int | None = typer.Option(None, "--patience", help="Early stopping patience"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed"),
) -> None:
    """Train a YOLO model for document segmentation."""
    setup_logging()
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

    settings = get_settings()
    data_dir = settings.paths.training_dir / "data" / data_subdir
    dataset_dir = data_dir / "dataset"
    project_dir = settings.paths.training_dir / "runs"

    logger.info("Starting YOLO training")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Project directory: {project_dir}")

    trainer = Trainer(
        dataset_dir=dataset_dir,
        project_dir=project_dir,
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
