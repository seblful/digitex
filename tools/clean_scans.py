"""Clean a directory of book scans before they enter the extraction pipeline.

Replaces the manual NAPS2 "document correction" pass the book archive was
built with: read every page in a directory, burn the gray paper out to white,
average the scanner grain away, cut off the scanner's white margin, and write
the page back out as PNG under the same stem.
"""

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import structlog
import typer
from PIL import Image

from digitex.config import get_settings
from digitex.domain.corpus import is_image, natural_sort_key
from digitex.imaging import correct_document
from digitex.logging import setup_logging

logger = structlog.get_logger()

app = typer.Typer(help="Whiten the paper in a directory of book scans")


@app.callback()
def configure() -> None:
    """Set up logging before the command runs."""
    setup_logging(get_settings())


def _clean_page(source: Path, output_dir: Path) -> Path:
    """Correct one page into *output_dir* — the unit of work a worker takes."""
    with Image.open(source) as page:
        dpi = page.info.get("dpi")
        cleaned = correct_document(page)
    target = output_dir / f"{source.stem}.png"
    cleaned.save(target, **({"dpi": dpi} if dpi else {}))
    return target


@app.command()
def clean_scans(
    input_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Directory of scanned pages"
    ),
    output_dir: Path = typer.Argument(..., help="Where the cleaned pages go"),
) -> None:
    """Clean every page in *input_dir* into *output_dir*.

    Each page is corrected against its own histogram, so a batch may hold
    scans of different paper, and pages go out to worker processes — the
    filter is a second of arithmetic per page and pages do not depend on
    each other.

    Args:
        input_dir: Directory of scanned pages.
        output_dir: Directory the cleaned PNGs are written to.
    """
    pages = sorted(
        (p for p in input_dir.iterdir() if is_image(p)), key=natural_sort_key
    )
    if not pages:
        typer.echo(f"No images found in {input_dir}")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor() as pool:
        for target in pool.map(_clean_page, pages, repeat(output_dir)):
            logger.info("cleaned_page", target=str(target))

    typer.echo(f"Cleaned {len(pages)} pages into {output_dir}")


if __name__ == "__main__":
    app()
