"""Run extraction of question images from PDF books."""

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
def extract(preprocess: str | None = "enhance") -> None:
    """Extract question images from all PDF books.

    Args:
        preprocess: Preprocessing mode: None, "enhance", "binarize", "grabcut", or "threshold".
    """
    settings = get_settings()
    extractor = TestsExtractor(
        model_path=settings.extraction.model_path,
        render_scale=settings.extraction.render_scale,
        image_format=settings.extraction.image_format,
        books_dir=settings.extraction.books_dir,
        extraction_dir=settings.extraction.extraction_dir,
        preprocess=preprocess,
    )
    extractor.extract_all()


if __name__ == "__main__":
    app()
