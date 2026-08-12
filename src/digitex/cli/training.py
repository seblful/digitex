"""Training CLI commands.

Settings are resolved per command rather than at import, so importing this
module reads no files and configures no logging. The heavy imports
(ultralytics, torch) stay inside the commands that need them.
"""

from pathlib import Path

import typer

from digitex.config import Settings, get_settings
from digitex.logging import setup_logging

app = typer.Typer(help="YOLO model training for document segmentation.")


@app.callback()
def configure() -> None:
    """Set up logging before any command runs."""
    setup_logging(get_settings())


def _data_dir(settings: Settings, data_type_dir_name: str) -> Path:
    return settings.paths.training_data_dir / data_type_dir_name


def _abort(message: str) -> typer.Exit:
    """Render *message* on stderr and return the exit to raise."""
    typer.echo(typer.style(message, fg="red", bold=True), err=True)
    return typer.Exit(code=1)


@app.command(name="create-dataset")
def create_dataset(
    data_type_dir_name: str = typer.Argument(
        ..., help="Data type subdirectory name (e.g. page)"
    ),
    train_split: float = typer.Option(
        0.8, "--train-split", help="Fraction of data used for training"
    ),
) -> None:
    """Convert Label Studio annotations into a YOLO training dataset."""
    from digitex.ml.yolo.dataset import DatasetCreator

    settings = get_settings()
    data_dir = _data_dir(settings, data_type_dir_name)
    annotations_file = data_dir / "annotations.json"
    images_dir = data_dir / settings.data.images_dir_name
    dataset_dir = data_dir / settings.data.dataset_dir_name

    if not annotations_file.exists():
        raise _abort(f"Error: annotations file not found: {annotations_file}")

    dataset = DatasetCreator(
        annotations_file=annotations_file,
        images_dir=images_dir,
        dataset_dir=dataset_dir,
        train_split=train_split,
    ).create()

    typer.echo(
        typer.style(
            f"✓ Dataset created at {dataset.dataset_dir}:"
            f" {dataset.train} train, {dataset.val} val, {dataset.test} test",
            fg="green",
        )
    )
    if dataset.missing_images:
        typer.echo(
            typer.style(
                f"  {len(dataset.missing_images)} annotated image(s) not found"
                f" in {images_dir}",
                fg="yellow",
            )
        )


@app.command(name="add-images")
def add_images(
    data_type_dir_name: str = typer.Argument(
        "page", help="Data type subdirectory name (e.g. page, question, part)"
    ),
) -> None:
    """Add images listed in images.txt to training data."""
    from digitex.creators import PageDataCreator

    settings = get_settings()
    data_dir = _data_dir(settings, data_type_dir_name)
    paths_file = data_dir / "images.txt"

    if not paths_file.exists():
        raise _abort(f"Error: {paths_file} not found")

    if not paths_file.read_text(encoding="utf-8").strip():
        typer.echo("images.txt is empty.")
        raise typer.Exit(code=0)

    output_dir = data_dir / settings.data.images_dir_name
    PageDataCreator(image_size=settings.data.image_size).add_from_file(
        paths_file=paths_file,
        output_dir=output_dir,
    )
    typer.echo(typer.style(f"✓ Images added to {output_dir}", fg="green"))


@app.command(name="select-random-pages")
def select_random_pages(
    num_images: int = typer.Option(
        100, "--num-images", help="Number of page images to sample"
    ),
) -> None:
    """Randomly sample page images from the books directory for training."""
    from digitex.creators import PageDataCreator

    settings = get_settings()
    page_train_dir = _data_dir(settings, "page") / settings.data.images_dir_name

    PageDataCreator(image_size=settings.data.image_size).create(
        books_dir=settings.paths.books_dir,
        output_dir=page_train_dir,
        num_images=num_images,
    )
    typer.echo(
        typer.style(
            f"✓ Selected {num_images} random pages into {page_train_dir}", fg="green"
        )
    )


@app.command(name="train")
def train(
    config: str = typer.Option(
        "page",
        "--config",
        help="Config base name (expects {config}_train.yaml and {config}_val.yaml)",
    ),
) -> None:
    """Train and validate a YOLO segmentation model."""
    from digitex.ml.yolo import training

    configs_dir = get_settings().paths.training_configs_dir

    try:
        training.run(
            train_config=configs_dir / f"{config}_train.yaml",
            val_config=configs_dir / f"{config}_val.yaml",
        )
    except FileNotFoundError as exc:
        raise _abort(f"Error: config not found: {exc}") from None
    except ValueError as exc:
        raise _abort(f"Error: {exc}") from None

    typer.echo(typer.style("✓ Training and validation completed", fg="green"))


@app.command(name="ls-predict")
def ls_predict(
    project_id: int = typer.Option(..., "--project-id", help="Label Studio project ID"),
    model_path: str = typer.Option(
        ..., "--model-path", help="Path to trained YOLO model (.pt file)"
    ),
) -> None:
    """Run model predictions on Label Studio tasks for a project."""
    from digitex.label_studio import TaskPredictor

    settings = get_settings()
    predictor = TaskPredictor(
        model_path=model_path,
        url=settings.label_studio.url,
        api_key=settings.label_studio.api_key,
    )

    count = predictor.predict_tasks(project_id)
    typer.echo(
        typer.style(f"✓ Predicted {count} tasks in project {project_id}", fg="green")
    )


if __name__ == "__main__":
    app()
