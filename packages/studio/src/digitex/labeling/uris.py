"""Where Label Studio says an image is.

Two pieces of the tool's vocabulary and nothing else: the
``/data/local-files/?d=...`` (or ``?file=...``) URI it serves a local image
through, and the fact that the column holding one is named differently
depending on how the task was imported. Both are Label Studio's spellings, so
they live with the rest of what this package knows about Label Studio.

Deliberately not here: anything that resolves a path against a document root or
checks whether the file exists. This module reads what the URI says, and the
caller decides what that means on its machine.

It used to sit in ``domain``, because the annotation client and the YOLO
dataset builder both needed it and homing it in either would have made the two
packages import each other. That was only true while the dataset builder read
the tool's export format itself. It takes
:class:`~digitex.domain.annotations.AnnotatedImage` now, so nothing under ``ml``
has any reason to know what a local-files URI looks like.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urlparse

if TYPE_CHECKING:
    from collections.abc import Mapping

_IMAGE_DATA_KEY = "image"

# The two spellings of the same parameter. Which one a URI carries depends on
# the Label Studio version that wrote it, never on the caller.
_PATH_PARAMS = ("file", "d")


def local_file_path(image_uri: str) -> Path | None:
    """Read the local filesystem path out of a ``local-files`` URI.

    Args:
        image_uri: A URI as Label Studio stores it, of the form
            ``/data/local-files/?d=...`` or ``/data/local-files/?file=...``.

    Returns:
        The path the URI names, or None when *image_uri* is empty or carries no
        local-file parameter at all — a remote URL, or a blob-storage link.
    """
    if not image_uri:
        return None

    params = parse_qs(urlparse(image_uri).query)
    for param in _PATH_PARAMS:
        if param in params:
            # The separators in the URI are whatever the Label Studio host
            # indexed its files with — backslashes from a Windows server — and
            # say nothing about the machine reading it. ``PureWindowsPath``
            # splits on both kinds, so the result is the same here, in CI and
            # in a container. ``url2pathname`` is not: it left a Windows URI as
            # one long filename on Linux.
            return Path(PureWindowsPath(unquote(params[param][0])).as_posix())
    return None


def task_image_path(data: Mapping[str, Any]) -> Path | None:
    """The local path of a task's image, whichever column holds it.

    ``image`` is the column the label config names, and the one an import of a
    file of paths writes. A sync from a storage of blob URLs writes
    ``$undefined$`` instead — Label Studio names the column that when the
    import carries no field name of its own, and resolves it against the single
    object tag at render time. A reader that only knows ``image`` passes over
    every task of such a project and says nothing about why.

    Args:
        data: A task's ``data`` mapping, as the SDK returns it.

    Returns:
        The path of the first column that parses as a local-files URI, or None
        when no column does.
    """
    # ``image`` first, then the rest in the order the task carries them: when
    # both columns are present one of them is a leftover, and it is not the one
    # the label config names.
    others = (key for key in data if key != _IMAGE_DATA_KEY)
    for key in (_IMAGE_DATA_KEY, *others):
        value = data.get(key)
        if isinstance(value, str) and (path := local_file_path(value)) is not None:
            return path
    return None
