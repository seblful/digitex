import logging
from pathlib import Path

import typer

from digitex.creators import PageDataCreator
from digitex.label_studio import TaskPredictor
from digitex.ml.yolo import Trainer
from digitex.ml.yolo.dataset import DatasetCreator

app = typer.Typer(help="YOLO model training for document segmentation")
logger = logging.getLogger(__name__)


def _data_dir(data_type_dir_name: str) -> Path:
    from digitex.config import get_settings

    s = get_settings()
    return s.paths.training_dir / s.data.data_dir_name / data_type_dir_name


@app.command()
def create_dataset(
    data_type_dir_name: str = typer.Argument(..., help="Type of task type"),
    train_split: float = typer.Option(0.8, "--train-split", help="Split of train set"),
) -> None:
    from digitex.config import get_settings

    s = get_settings()
    data_dir = _data_dir(data_type_dir_name)
    annotations_file = data_dir / "annotations.json"
    images_dir = data_dir / s.data.images_dir_name
    dataset_dir = data_dir / s.data.dataset_dir_name

    creator = DatasetCreator(
        annotations_file=annotations_file,
        images_dir=images_dir,
        dataset_dir=dataset_dir,
        train_split=train_split,
    )
    creator.create()


@app.command()
def select_random_pages(
    num_images: int = typer.Option(
        100, "--num-images", help="Number of images to create"
    ),
) -> None:
    from digitex.config import get_settings

    s = get_settings()
    page_train_dir = _data_dir("page") / s.data.images_dir_name

    PageDataCreator(image_size=s.data.image_size).create(
        books_dir=s.paths.books_dir,
        output_dir=page_train_dir,
        num_images=num_images,
    )


@app.command()
def train(
    config: str = typer.Option("page", "--config", help="Config name (without .yaml)"),
) -> None:
    from digitex.config import get_settings

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    s = get_settings()
    config_path = s.paths.training_dir / s.training.configs_dir_name / f"{config}.yaml"

    logger.info("Starting YOLO training")
    logger.info(f"Using config: {config_path}")

    try:
        trainer = Trainer(
            config_path=config_path,
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

    if not model_path:
        typer.echo("Error: No model path. Use --model-path.")
        raise typer.Exit(1)

    predictor = TaskPredictor(
        model_path=model_path,
        url=s.label_studio.url,
        api_key=s.label_studio.api_key,
    )

    count = predictor.predict_tasks(project_id)
    typer.echo(f"Predicted {count} tasks.")


if __name__ == "__main__":
    app()
