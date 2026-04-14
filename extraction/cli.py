"""Run extraction of question images from image books."""

import platform
import shutil
import tempfile
from pathlib import Path

if platform.system() == "Windows":
    import pathlib
    import pathlib._local as _local

    _local.PosixPath = pathlib.WindowsPath

import typer

from digitex import TestsExtractor
from digitex.config import get_settings
from digitex.logging import setup_logging

setup_logging()

app = typer.Typer()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@app.command()
def extract() -> None:
    """Extract question images from all image books."""
    settings = get_settings()
    model_path = settings.paths.home_dir / settings.extraction.model_path
    extractor = TestsExtractor(
        model_path=model_path,
        image_format=settings.extraction.image_format,
        question_max_width=settings.extraction.question_max_width,
        question_max_height=settings.extraction.question_max_height,
        books_dir=settings.paths.books_dir,
        extraction_dir=settings.paths.extraction_dir
        / settings.extraction.output_dir_name,
    )
    extractor.extract_all()


@app.command()
def renumber(
    dry_run: bool = typer.Option(True, help="Preview changes without renaming"),
) -> None:
    """Renumber images in the extraction output folder to fill gaps (e.g., 1, 2, 4, 5 -> 1, 2, 3, 4)."""
    settings = get_settings()
    folder = settings.paths.extraction_dir / settings.extraction.output_dir_name

    if not folder.exists() or not folder.is_dir():
        typer.echo(f"Error: {folder} is not a valid directory")
        raise typer.Exit(code=1)

    def find_image_folders(root: Path) -> list[Path]:
        result = []
        for item in root.iterdir():
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
                return [root]
            if item.is_dir():
                result.extend(find_image_folders(item))
        return result

    total = 0
    for fp in find_image_folders(folder):
        images = sorted(
            (int(f.stem), f)
            for f in fp.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            continue

        current = [n for n, _ in images]
        expected = list(range(1, len(images) + 1))
        if current == expected:
            continue

        changes = [
            (f, f.parent / f"{i}{f.suffix}")
            for i, (_, f) in enumerate(images, 1)
            if f.stem != str(i)
        ]

        rel = fp.relative_to(folder)
        if dry_run:
            typer.echo(f"{rel}:")
            for o, n in changes:
                typer.echo(f"  {o.name} -> {n.name}")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                tp = Path(tmp)
                for old, new in changes:
                    shutil.move(str(old), str(tp / new.name))
                    shutil.move(str(tp / new.name), str(new))
        total += len(changes)

    if dry_run and total:
        typer.echo(f"\n{total} files would be renamed")
    elif total:
        typer.echo(f"Renamed {total} files successfully")
    else:
        typer.echo("All images are already sequential")


if __name__ == "__main__":
    app()
