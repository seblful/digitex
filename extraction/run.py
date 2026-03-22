"""Run extraction of question images from PDF books."""

import logging

import typer
from digitex import Extractor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
)

app = typer.Typer()


@app.command()
def extract() -> None:
    """Extract question images from all PDF books."""
    extractor = Extractor()
    extractor.extract_all()


if __name__ == "__main__":
    app()
