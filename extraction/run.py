"""Run extraction of question images from image books."""

import logging

import typer
from digitex import TestsExtractor
from digitex.config import get_settings

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
)

app = typer.Typer()


@app.command()
def extract() -> None:
    """Extract question images from all image books."""
    settings = get_settings()
    extractor = TestsExtractor(
        model_path=settings.extraction.model_path,
        image_format=settings.extraction.image_format,
        question_max_width=settings.extraction.question_max_width,
        question_max_height=settings.extraction.question_max_height,
        books_dir=settings.extraction.books_dir,
        extraction_dir=settings.extraction.extraction_dir,
    )
    extractor.extract_all()


if __name__ == "__main__":
    app()
