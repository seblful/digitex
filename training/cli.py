from collections import Counter
from pathlib import Path

import structlog
import typer

from digitex.creators import PageDataCreator
from digitex.extractors.manual_extractor import ManualExtractor
from digitex.label_studio import TaskPredictor
from digitex.ml.yolo import Trainer
from digitex.ml.yolo.dataset import DatasetCreator

app = typer.Typer(help="YOLO model training for document segmentation")
logger = structlog.get_logger()


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
def add_images(
    data_type_dir_name: str = typer.Argument(
        "page", help="Type of data (e.g., page, question, part)"
    ),
) -> None:
    """Add specific images from paths.txt to training data."""
    from digitex.config import get_settings

    s = get_settings()
    data_dir = _data_dir(data_type_dir_name)
    paths_file = data_dir / "images.txt"

    if not paths_file.exists():
        typer.echo(f"Error: {paths_file} not found.")
        raise typer.Exit(code=1)

    if not paths_file.read_text(encoding="utf-8").strip():
        typer.echo("images.txt is empty.")
        raise typer.Exit(code=0)

    output_dir = data_dir / s.data.images_dir_name

    PageDataCreator(image_size=s.data.image_size).add_from_file(
        paths_file=paths_file,
        output_dir=output_dir,
    )


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
def count(
    subject: str | None = typer.Option(
        None, "--subject", help="Subject to count (e.g., biology)"
    ),
    year: int | None = typer.Option(None, "--year", help="Year to filter"),
    option: int | None = typer.Option(None, "--option", help="Option number to filter"),
    part: str | None = typer.Option(
        None, "--part", help="Part letter (A or B)"
    ),
) -> None:
    """Count extracted images by subject/year/option/part."""

    from digitex.config import get_settings

    s = get_settings()
    extraction_output = s.paths.extraction_dir / s.extraction.output_dir_name

    if not extraction_output.exists():
        typer.echo(f"Extraction output not found: {extraction_output}")
        raise typer.Exit(code=1)

    if subject:
        subject_dirs = [extraction_output / subject]
        if not subject_dirs[0].exists():
            typer.echo(f"Subject not found: {subject}")
            raise typer.Exit(code=1)
    else:
        subject_dirs = [d for d in extraction_output.iterdir() if d.is_dir()]

    if not subject_dirs:
        typer.echo("No subjects found.")
        raise typer.Exit(code=0)

    total = 0
    for subject_dir in subject_dirs:
        subject_counts: Counter = Counter()
        for year_dir in subject_dir.iterdir():
            if not year_dir.is_dir():
                continue
            if year and year_dir.name != str(year):
                continue
            for option_dir in year_dir.iterdir():
                if not option_dir.is_dir():
                    continue
                if option and option_dir.name != str(option):
                    continue
                for part_dir in option_dir.iterdir():
                    if not part_dir.is_dir():
                        continue
                    if part and part_dir.name != part:
                        continue
                    count = len(list(part_dir.glob("*.jpg")))
                    if count > 0:
                        key = f"{subject_dir.name}/{year_dir.name}/{option_dir.name}/{part_dir.name}"
                        subject_counts[key] = count
                        total += count

        if subject_counts:
            typer.echo(f"\n{subject_dir.name}:")
            for key, count in sorted(subject_counts.items()):
                typer.echo(f"  {key}: {count}")

    if not subject:
        typer.echo(f"\nTotal: {total}")


@app.command()
def renumber(
    subject: str = typer.Option(..., "--subject", help="Subject (e.g., biology)"),
    year: int = typer.Option(..., "--year", help="Year"),
    option: int = typer.Option(..., "--option", help="Option number"),
    part: str = typer.Option(..., "--part", help="Part letter (A or B)"),
    start: int = typer.Option(..., "--start", help="Start renumbering from this question"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without applying"
    ),
) -> None:
    """Renumber existing question images starting from a specific number."""

    from digitex.config import get_settings

    s = get_settings()
    extractor = ManualExtractor(
        image_format=s.extraction.image_format,
        question_max_width=s.extraction.question_max_width,
        question_max_height=s.extraction.question_max_height,
        manual_dir=s.paths.extraction_dir / "data",
        output_dir=s.paths.extraction_dir / "output",
    )

    target_dir = (
        s.paths.extraction_dir
        / "output"
        / subject
        / str(year)
        / str(option)
        / part
    )

    if not target_dir.exists():
        typer.echo(f"Directory not found: {target_dir}")
        raise typer.Exit(code=1)

    action = "Would renumber" if dry_run else "Renumbering"
    typer.echo(f"{action} files in {target_dir} starting from {start}")

    changes = extractor._renumber_files(target_dir, start, dry_run=dry_run)

    if not changes:
        typer.echo("No files to renumber.")
        return

    for old_path, new_path in changes:
        status = "[DRY RUN]" if dry_run else "Renamed"
        typer.echo(f"  {status}: {old_path.name} -> {new_path.name}")

    if dry_run:
        typer.echo("\nRun without --dry-run to apply changes.")
    else:
        typer.echo(f"\nRenumbered {len(changes)} files.")


@app.command()
def extract_manual(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without applying"
    ),
) -> None:
    """Process manually cropped question images."""
    from digitex.config import get_settings

    s = get_settings()
    extractor = ManualExtractor(
        image_format=s.extraction.image_format,
        question_max_width=s.extraction.question_max_width,
        question_max_height=s.extraction.question_max_height,
        manual_dir=s.paths.extraction_dir / "data",
        output_dir=s.paths.extraction_dir / "output",
    )

    extractor.process_all(dry_run=dry_run)


@app.command()
def train(
    config: str = typer.Option(
        "page",
        "--config",
        help="Train and val config name (without _train.yaml, _val.yaml)",
    ),
) -> None:
    from digitex.config import get_settings
    from digitex.logging import setup_logging

    setup_logging()

    s = get_settings()
    train_config_path = (
        s.paths.training_dir / s.training.configs_dir_name / f"{config}_train.yaml"
    )
    val_config_path = (
        s.paths.training_dir / s.training.configs_dir_name / f"{config}_val.yaml"
    )

    logger.info("Starting YOLO training")
    logger.info(f"Using train config: {train_config_path}")
    logger.info(f"Using val config: {val_config_path}")

    try:
        trainer = Trainer(
            train_config_path=train_config_path,
            val_config_path=val_config_path,
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
