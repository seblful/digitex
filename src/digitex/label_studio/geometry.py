"""Label Studio geometry — local-file URIs and the percent point space.

Label Studio references local images as ``/data/local-files/?d=...`` (or
``?file=...``) URIs and stores polygon points as percentages (0-100) of the
image size. Parsing those URIs, and every conversion into or out of the percent
space, happens here. The spaces themselves are named in
:mod:`digitex.core.domain`; scaling a YOLO mask up to pixels belongs to
:mod:`digitex.ml.predictors`, which is where masks come from.
"""

from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import url2pathname

from digitex.core.domain import NormalizedPolygon, PercentPolygon, PixelPolygon


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


def percent_to_normalized(points: PercentPolygon) -> NormalizedPolygon:
    """Convert Label Studio percent points (0-100) to normalized (0-1)."""
    return NormalizedPolygon([(x / 100, y / 100) for x, y in points])


def pixel_to_percent(
    polygon: PixelPolygon, img_width: int, img_height: int
) -> PercentPolygon:
    """Convert pixel points to Label Studio percent points (0-100)."""
    return PercentPolygon(
        [[x / img_width * 100, y / img_height * 100] for x, y in polygon]
    )
