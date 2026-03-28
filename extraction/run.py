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
def extract(
    preprocess: str | None = "threshold",
    bg_threshold: int = 200,
) -> None:
    """Extract question images from all PDF books.

    Args:
        preprocess: Preprocessing mode: None or "threshold" (removes white background).
        bg_threshold: Threshold for background removal (0-255, lower = more transparent).
    """
    settings = get_settings()
    extractor = TestsExtractor(
        model_path=settings.extraction.model_path,
        render_scale=settings.extraction.render_scale,
        image_format=settings.extraction.image_format,
        books_dir=settings.extraction.books_dir,
        extraction_dir=settings.extraction.extraction_dir,
        preprocess=preprocess,
        bg_threshold=bg_threshold,
    )
    extractor.extract_all()


if __name__ == "__main__":
    app()
