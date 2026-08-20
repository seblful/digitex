"""Training CLI commands.

Settings are resolved per command rather than at import, so importing this module
reads no files and configures no logging. The heavy imports — ultralytics, torch —
stay inside the commands that need them, which is what keeps `--help` fast on a
machine with no GPU.
"""

from pathlib import Path

import typer

from digitex.config import Settings, get_settings
from digitex.console import abort
from digitex.domain.corpus import book_subjects
from digitex.logging import setup_logging

app = typer.Typer(help="YOLO model training for document segmentation.")


@app.callback()
def configure() -> None:
    """Set up logging before any command runs."""
    setup_logging(get_settings())


def _ok(message: str) -> None:
    """Report a finished command."""
    typer.echo(typer.style(message, fg="green"))


def _note(message: str) -> None:
    """Report something the operator should look at but that failed nothing."""
    typer.echo(typer.style(message, fg="yellow"))


def _data_dir(settings: Settings, data_type_dir_name: str) -> Path:
    return settings.paths.training_data_dir / data_type_dir_name


@app.command(name="create-dataset")
def create_dataset(
    data_type_dir_name: str = typer.Argument(
        ..., help="Data type subdirectory name (e.g. page)"
    ),
    train_split: float = typer.Option(
        0.8,
        "--train-split",
        min=0.0,
        max=1.0,
        help="Fraction of data used for training",
    ),
) -> None:
    """Convert Label Studio annotations into a YOLO training dataset."""
    from digitex.labeling.export import read_export
    from digitex.ml.yolo.dataset import DatasetCreator

    settings = get_settings()
    data_dir = _data_dir(settings, data_type_dir_name)
    annotations_file = data_dir / "annotations.json"
    images_dir = data_dir / settings.pipeline.data.images_dir_name
    dataset_dir = data_dir / settings.pipeline.data.dataset_dir_name

    if not annotations_file.exists():
        raise abort(f"Error: annotations file not found: {annotations_file}")

    # Where the two halves meet: the annotation tool's format is read here and
    # the trainer is handed plain annotations, so neither knows the other.
    dataset = DatasetCreator(
        annotations=read_export(annotations_file),
        images_dir=images_dir,
        dataset_dir=dataset_dir,
        train_split=train_split,
    ).create()

    _ok(
        f"✓ Dataset created at {dataset.dataset_dir}:"
        f" {dataset.train} train, {dataset.val} val, {dataset.test} test"
    )
    if dataset.missing_images:
        _note(
            f"  {len(dataset.missing_images)} annotated image(s) not found"
            f" in {images_dir}"
        )


@app.command(name="add-images")
def add_images(
    data_type_dir_name: str = typer.Argument(
        "page", help="Data type subdirectory name (e.g. page, question, part)"
    ),
) -> None:
    """Add images listed in images.txt to training data."""
    from digitex.pipeline.training_pool import PageDataCreator

    settings = get_settings()
    data_dir = _data_dir(settings, data_type_dir_name)
    paths_file = data_dir / "images.txt"

    if not paths_file.exists():
        raise abort(f"Error: {paths_file} not found")

    if not paths_file.read_text(encoding="utf-8").strip():
        typer.echo("images.txt is empty.")
        raise typer.Exit(code=0)

    output_dir = data_dir / settings.pipeline.data.images_dir_name
    PageDataCreator(image_size=settings.pipeline.data.image_size).add_from_file(
        paths_file=paths_file,
        output_dir=output_dir,
    )
    _ok(f"✓ Images added to {output_dir}")


@app.command(name="select-random-pages")
def select_random_pages(
    subject: str | None = typer.Argument(
        None, help="Subject to sample; omit to sample every subject"
    ),
    num_images: int = typer.Option(
        100, "--num-images", help="Number of page images to sample"
    ),
) -> None:
    """Randomly sample page images from the books directory for training."""
    from digitex.pipeline.training_pool import PageDataCreator

    settings = get_settings()
    data = settings.pipeline.data
    page_train_dir = _data_dir(settings, "page") / data.images_dir_name

    # A typo would otherwise read as "this subject has no pages" and abort with
    # the same message as an empty archive.
    subjects = book_subjects(settings.paths.books_dir)
    if subject is not None and subject not in subjects:
        raise abort(
            f"Error: unknown subject {subject!r};"
            f" the archive holds: {', '.join(subjects) or 'nothing'}"
        )

    PageDataCreator(image_size=data.image_size).create(
        books_dir=settings.paths.books_dir,
        output_dir=page_train_dir,
        num_images=num_images,
        subject=subject,
    )
    _ok(
        f"✓ Sampled up to {num_images} pages"
        f" from {subject or 'every subject'} into {page_train_dir}"
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
        raise abort(f"Error: config not found: {exc}") from None
    except ValueError as exc:
        raise abort(f"Error: {exc}") from None

    _ok("✓ Training and validation completed")


if __name__ == "__main__":
    app()
