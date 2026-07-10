"""Label Studio geometry — local-file URIs and polygon point conversions.

Label Studio references local images as ``/data/local-files/?d=...`` (or
``?file=...``) URIs and stores polygon points as percentages (0-100) of the
image size. Parsing those URIs and converting between the percent space and
the pipeline's other coordinate spaces happens only here.
"""

from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import url2pathname


def local_file_path(image_uri: str) -> Path | None:
    """Extract the local filesystem path from a local-files URI.

    Handles URIs of the form ``/data/local-files/?d=...`` and
    ``/data/local-files/?file=...``. Returns None when the URI is empty or
    has no local-file parameter.
    """
    if not image_uri:
        return None

    params = parse_qs(urlparse(image_uri).query)
    for key in ("file", "d"):
        if key in params:
            return Path(url2pathname(params[key][0]))
    return None


def percent_to_normalized(
    points: list[list[float]],
) -> list[tuple[float, float]]:
    """Convert Label Studio percent points (0-100) to normalized (0-1)."""
    return [(x / 100, y / 100) for x, y in points]


def pixel_to_percent(
    polygon: list[tuple[int, int]], img_width: int, img_height: int
) -> list[list[float]]:
    """Convert pixel points to Label Studio percent points (0-100)."""
    return [[x / img_width * 100, y / img_height * 100] for x, y in polygon]
