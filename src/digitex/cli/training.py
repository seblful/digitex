"""Training CLI commands."""

import structlog
import typer

from digitex.config import get_settings
from digitex.logging import setup_logging

setup_logging()

app = typer.Typer(help="YOLO model training for document segmentation.")
logger = structlog.get_logger()


def _data_dir(data_type_dir_name: str):
    s = get_settings()
    return s.paths.training_data_dir / data_type_dir_name


@app.command(name="create-dataset")
def create_dataset(
    data_type_dir_name: str = typer.Argument(..., help="Data type subdirectory name (e.g. page)"),
    train_split: float = typer.Option(0.8, "--train-split", help="Fraction of data used for training"),
) -> None:
    """Convert Label Studio annotations into a YOLO training dataset."""
    from digitex.ml.yolo.dataset import DatasetCreator

    s = get_settings()
    data_dir = _data_dir(data_type_dir_name)
    annotations_file = data_dir / "annotations.json"
    images_dir = data_dir / s.data.images_dir_name
    dataset_dir = data_dir / s.data.dataset_dir_name

    if not annotations_file.exists():
        typer.echo(typer.style(f"Error: annotations file not found: {annotations_file}", fg="red"), err=True)
        raise typer.Exit(code=1)

    creator = DatasetCreator(
        annotations_file=annotations_file,
        images_dir=images_dir,
        dataset_dir=dataset_dir,
        train_split=train_split,
    )
    creator.create()
    typer.echo(typer.style(f"✓ Dataset created at {dataset_dir}", fg="green"))


@app.command(name="add-images")
def add_images(
    data_type_dir_name: str = typer.Argument(
        "page", help="Data type subdirectory name (e.g. page, question, part)"
    ),
) -> None:
    """Add images listed in images.txt to training data."""
    from digitex.creators import PageDataCreator

    s = get_settings()
    data_dir = _data_dir(data_type_dir_name)
    paths_file = data_dir / "images.txt"

    if not paths_file.exists():
        typer.echo(typer.style(f"Error: {paths_file} not found", fg="red"), err=True)
        raise typer.Exit(code=1)

    if not paths_file.read_text(encoding="utf-8").strip():
        typer.echo("images.txt is empty.")
        raise typer.Exit(code=0)

    output_dir = data_dir / s.data.images_dir_name
    PageDataCreator(image_size=s.data.image_size).add_from_file(
        paths_file=paths_file,
        output_dir=output_dir,
    )
    typer.echo(typer.style(f"✓ Images added to {output_dir}", fg="green"))


@app.command(name="select-random-pages")
def select_random_pages(
    num_images: int = typer.Option(100, "--num-images", help="Number of page images to sample"),
) -> None:
    """Randomly sample page images from the books directory for training."""
    from digitex.creators import PageDataCreator

    s = get_settings()
    page_train_dir = _data_dir("page") / s.data.images_dir_name

    PageDataCreator(image_size=s.data.image_size).create(
        books_dir=s.paths.books_dir,
        output_dir=page_train_dir,
        num_images=num_images,
    )
    typer.echo(typer.style(f"✓ Selected {num_images} random pages into {page_train_dir}", fg="green"))


@app.command(name="train")
def train(
    config: str = typer.Option(
        "page",
        "--config",
        help="Config base name (expects {config}_train.yaml and {config}_val.yaml)",
    ),
) -> None:
    """Train and validate a YOLO segmentation model."""
    from digitex.ml.yolo import Trainer

    s = get_settings()
    configs_dir = s.paths.training_configs_dir
    train_config_path = configs_dir / f"{config}_train.yaml"
    val_config_path = configs_dir / f"{config}_val.yaml"

    if not train_config_path.exists():
        typer.echo(typer.style(f"Error: train config not found: {train_config_path}", fg="red"), err=True)
        raise typer.Exit(code=1)

    if not val_config_path.exists():
        typer.echo(typer.style(f"Error: val config not found: {val_config_path}", fg="red"), err=True)
        raise typer.Exit(code=1)

    logger.info("Starting YOLO training", train_config=str(train_config_path))

    try:
        trainer = Trainer(
            train_config_path=train_config_path,
            val_config_path=val_config_path,
        )
        trainer.train()
        trainer.validate()
        typer.echo(typer.style("✓ Training and validation completed", fg="green"))
    except Exception as e:
        logger.error("Training failed", error=str(e))
        typer.echo(typer.style(f"✗ Training failed: {e}", fg="red", bold=True), err=True)
        raise typer.Exit(code=1)


@app.command(name="ls-predict")
def ls_predict(
    project_id: int = typer.Option(..., "--project-id", help="Label Studio project ID"),
    model_path: str = typer.Option(..., "--model-path", help="Path to trained YOLO model (.pt file)"),
) -> None:
    """Run model predictions on Label Studio tasks for a project."""
    from digitex.label_studio import TaskPredictor

    s = get_settings()
    predictor = TaskPredictor(
        model_path=model_path,
        url=s.label_studio.url,
        api_key=s.label_studio.api_key,
    )

    count = predictor.predict_tasks(project_id)
    typer.echo(typer.style(f"✓ Predicted {count} tasks in project {project_id}", fg="green"))


if __name__ == "__main__":
    app()
