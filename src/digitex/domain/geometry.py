"""Annotation geometry — local-file URIs and the percent point space.

Label Studio references local images as ``/data/local-files/?d=...`` (or
``?file=...``) URIs and stores polygon points as percentages (0-100) of the
image size. Parsing those URIs, and every conversion into or out of the percent
space, happens here.

Lives in ``domain`` rather than ``labeling`` because both sides of the
annotation round trip need it — ``labeling.predictor`` converts pixels up to
percent when posting predictions, and ``ml.yolo.dataset`` converts percent down
to normalized when reading an export back. Homed in either one, the two
packages would import each other. It earns the place: the polygon spaces
themselves are named in :mod:`digitex.domain.entities` and nothing here imports
beyond them. Scaling a YOLO mask up to pixels belongs to
:mod:`digitex.ml.predictors`, which is where masks come from.
"""

from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from digitex.domain.entities import NormalizedPolygon, PercentPolygon, PixelPolygon

_IMAGE_DATA_KEY = "image"


def local_file_path(image_uri: str) -> Path | None:
    """Extract the local filesystem path from a local-files URI.

    Handles URIs of the form ``/data/local-files/?d=...`` and
    ``/data/local-files/?file=...``. Returns None when the URI is empty or
    has no local-file parameter.

    The separators in the URI are the ones the Label Studio host indexed its
    files with — backslashes from a Windows server — and have nothing to do
    with the machine reading it. ``PureWindowsPath`` accepts both kinds, so
    the split is the same here, in CI, and in a container; ``url2pathname``
    was not, and left a Windows URI as one long filename on Linux.
    """
    if not image_uri:
        return None

    params = parse_qs(urlparse(image_uri).query)
    for key in ("file", "d"):
        if key in params:
            return Path(PureWindowsPath(unquote(params[key][0])).as_posix())
    return None


def task_image_path(data: Mapping[str, Any]) -> Path | None:
    """The local filesystem path of a task's image, whichever key holds it.

    ``image`` is the key the label config names, and the one an import of a file
    of paths writes. A sync from a storage of blob URLs writes ``$undefined$``
    instead — Label Studio names the column that when the import carries no
    field name of its own, and resolves it against the single object tag when it
    renders the task. Reading only ``image`` passes over every task of such a
    project, and says nothing about why.
    """
    for key in sorted(data, key=lambda name: name != _IMAGE_DATA_KEY):
        value = data[key]
        if isinstance(value, str):
            path = local_file_path(value)
            if path is not None:
                return path
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
