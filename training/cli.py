import logging
from pathlib import Path

import typer

from digitex.creators import PageDataCreator
from digitex.label_studio import TaskPredictor
from digitex.ml.yolo import Trainer
from digitex.ml.yolo.augmenter import PolygonAugmenter
from digitex.ml.yolo.dataset import DatasetCreator
from digitex.ml.yolo.visualizer import PolygonVisualizer

app = typer.Typer(help="YOLO model training for document segmentation")
logger = logging.getLogger(__name__)

TRAINING_ROOT = Path(__file__).parent.parent


def _data_dir(data_type_dir_name: str) -> Path:
    from digitex.config import get_settings

    s = get_settings()
    return s.paths.training_dir / s.data.data_dir_name / data_type_dir_name


@app.command()
def create_dataset(
    data_type_dir_name: str = typer.Option(
        "page", "--data-subdir", help="Type of task type"
    ),
    train_split: float = typer.Option(0.8, "--train-split", help="Split of train set"),
    vis_images: int = typer.Option(20, "--vis-images", help="Images to visualize"),
    augment: bool = typer.Option(
        False, "--augment", help="Whether to augment train data"
    ),
    aug_images: int = typer.Option(
        100, "--aug-images", help="Augmented images to create"
    ),
) -> None:
    from digitex.config import get_settings

    s = get_settings()
    data_dir = _data_dir(data_type_dir_name)
    annotations_file = data_dir / "annotations.json"
    images_dir = data_dir / s.data.images_dir_name
    dataset_dir = data_dir / s.data.dataset_dir_name
    check_images_dir = data_dir / "check-images"

    creator = DatasetCreator(
        annotations_file=annotations_file,
        images_dir=images_dir,
        dataset_dir=dataset_dir,
        train_split=train_split,
    )
    creator.create()

    PolygonVisualizer(
        dataset_dir=str(dataset_dir),
        check_images_dir=str(check_images_dir),
    ).visualize(num_images=vis_images)

    if augment:
        PolygonAugmenter(classes=creator.classes, dataset_dir=str(dataset_dir)).augment(
            num_images=aug_images
        )


@app.command()
def select_random_pages(
    data_type_dir_name: str = typer.Option(
        "page", "--data-subdir", help="Type of task type"
    ),
    num_images: int = typer.Option(
        100, "--num-images", help="Number of images to create"
    ),
) -> None:
    from digitex.config import get_settings

    s = get_settings()
    page_train_dir = _data_dir(data_type_dir_name) / s.data.images_dir_name

    PageDataCreator(image_size=s.label_studio.image_size).create(
        books_dir=s.paths.books_dir,
        output_dir=page_train_dir,
        num_images=num_images,
    )


@app.command()
def train(
    data_type_dir_name: str | None = typer.Option(
        None, "--data-subdir", help="Task type"
    ),
    model_type: str | None = typer.Option(None, "--model-type", help="YOLO model type"),
    model_size: str | None = typer.Option(None, "--model-size", help="YOLO model size"),
    pretrained_model_path: str | None = typer.Option(
        None, "--pretrained-model-path", help="Model path"
    ),
    num_epochs: int | None = typer.Option(None, "--num-epochs", help="Training epochs"),
    image_size: int | None = typer.Option(
        None, "--image-size", help="Input image size"
    ),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Batch size"),
    overlap_mask: bool | None = typer.Option(
        None, "--overlap-mask", help="Mask overlap"
    ),
    patience: int | None = typer.Option(
        None, "--patience", help="Early stopping patience"
    ),
    seed: int | None = typer.Option(None, "--seed", help="Random seed"),
) -> None:
    from digitex.config import get_settings

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    s = get_settings()
    train_defaults, data_defaults = s.training, s.data
    data_type_dir_name = data_type_dir_name or data_defaults.data_type_dir_name

    data_dir = _data_dir(data_type_dir_name)
    dataset_dir = data_dir / data_defaults.dataset_dir_name
    project_dir = TRAINING_ROOT / train_defaults.runs_dir_name

    logger.info(f"Starting YOLO training | Data: {data_dir} | Dataset: {dataset_dir}")

    try:
        trainer = Trainer(
            dataset_dir=dataset_dir,
            project_dir=project_dir,
            model_type=model_type or train_defaults.model_type,
            model_size=model_size or train_defaults.model_size,
            pretrained_model_path=pretrained_model_path
            or train_defaults.pretrained_model_path,
            num_epochs=num_epochs or train_defaults.num_epochs,
            image_size=image_size or data_defaults.image_size,
            batch_size=batch_size or train_defaults.batch_size,
            overlap_mask=overlap_mask
            if overlap_mask is not None
            else train_defaults.overlap_mask,
            patience=patience or train_defaults.patience,
            seed=seed or train_defaults.seed,
        )
        trainer.train()
        trainer.validate()
        logger.info("Training and validation completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def ls_predict(
    project_id: int = typer.Option(..., help="Label Studio project ID"),
    model_path: str | None = typer.Option(
        None,
        help="Path to trained model (.pt file)",
    ),
) -> None:
    from digitex.config import get_settings

    s = get_settings()

    model = model_path or s.training.pretrained_model_path
    if not model:
        typer.echo(
            "Error: No model path. Use --model-path or set TRAIN_PRETRAINED_MODEL_PATH."
        )
        raise typer.Exit(1)

    predictor = TaskPredictor(
        model_path=model,
        url=s.label_studio.url,
        api_key=s.label_studio.api_key,
    )

    count = predictor.predict_tasks(project_id)
    typer.echo(f"Predicted {count} tasks.")


if __name__ == "__main__":
    app()
