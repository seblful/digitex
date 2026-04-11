"""Run extraction of question images from image books."""

import platform

if platform.system() == "Windows":
    import pathlib
    import pathlib._local as _local

    _local.PosixPath = pathlib.WindowsPath

import structlog
import typer

from pathlib import Path

from digitex import TestsExtractor
from digitex.config import get_settings
from digitex.logging import setup_logging

setup_logging()

app = typer.Typer()


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


if __name__ == "__main__":
    app()
