"""Where Label Studio says an image is.

The tool references local images as ``/data/local-files/?d=...`` (or
``?file=...``) URIs, and names the column holding one differently depending on
how the task was imported. Reading either is knowledge about Label Studio, so
it lives with the rest of what this package knows about Label Studio.

It used to sit in ``domain`` on the grounds that both the annotation client and
the YOLO dataset builder needed it, and homing it in either would have made the
two packages import each other. That was true only because the dataset builder
read the tool's export format directly. It takes
:class:`~digitex.domain.annotations.AnnotatedImage` now, so nothing under
``ml`` has a reason to know what a local-files URI looks like.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse

if TYPE_CHECKING:
    from collections.abc import Mapping

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
