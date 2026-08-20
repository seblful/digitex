"""Reading a Label Studio export into annotations anything can train on.

Every assumption about the tool's JSON is in this file: that an entry names its
image under ``image``, that a region carries its class in ``polygonlabels`` and
its outline in ``points``, and that those points are percentages of the image
size. What comes out the other side is
:class:`~digitex.domain.annotations.AnnotatedImage` — a filename and normalized
polygons — which the YOLO dataset builder used to derive from the raw export
itself.

A partially malformed export still yields a usable dataset. A region missing a
label or its points is dropped with a warning rather than failing the read: the
alternative is one bad polygon in a batch of six hundred images costing a
training run, and the warning is what makes it findable.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import structlog

from digitex.domain.annotations import AnnotatedImage, LabelledRegion
from digitex.domain.geometry import percent_to_normalized
from digitex.labeling.uris import local_file_path

if TYPE_CHECKING:
    from pathlib import Path

    from digitex.domain.entities import PercentPolygon

logger = structlog.get_logger()


def _regions(entry: dict[str, Any]) -> tuple[LabelledRegion, ...]:
    """Every usable region on one entry, in the order the export lists them."""
    regions: list[LabelledRegion] = []

    for polygon in entry.get("label", []):
        try:
            label = polygon["polygonlabels"][0]
            # The one hop where untrusted export JSON is asserted to be Label
            # Studio's percent space.
            points = cast("PercentPolygon", polygon["points"])
            normalized = percent_to_normalized(points)
        except (KeyError, IndexError) as exc:
            logger.warning("skipped_polygon", reason=str(exc), polygon=polygon)
            continue

        # An empty points list raises nothing, and would emit a class id with
        # no coordinates — an invalid YOLO instance.
        if not normalized:
            logger.warning("skipped_polygon", reason="no points", polygon=polygon)
            continue

        regions.append(LabelledRegion(label=label, polygon=normalized))

    return tuple(regions)


def read_export(path: Path) -> list[AnnotatedImage]:
    """Every annotated image in the export at *path*.

    Entries whose image URI names no local file are dropped with a warning —
    a task synced from blob storage has no page on this disk to train on.

    Duplicate filenames are returned as they are found rather than resolved
    here: the export addresses images by URI and only the basename survives, so
    two annotation batches can each hold a ``30.jpg``. Deciding what to do
    about that belongs to whoever is assembling a dataset out of them.
    """
    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    images: list[AnnotatedImage] = []
    for entry in entries:
        local = local_file_path(entry["image"])
        if local is None:
            logger.warning("skipped_entry_no_local_path", image=entry.get("image"))
            continue
        images.append(AnnotatedImage(filename=local.name, regions=_regions(entry)))

    logger.info("read_export", images=len(images), path=str(path))
    return images
