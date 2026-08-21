"""Reading a Label Studio export into annotations anything can train on.

This file is where the tool's export JSON stops being anybody else's problem.
It is the only place that knows an entry names its image under ``image``, that
a region carries its class in ``polygonlabels`` and its outline in ``points``,
and that those points are percentages of the image size. Out the other side
comes :class:`~digitex.domain.annotations.AnnotatedImage` — a filename and
normalized polygons — which is all the YOLO dataset builder ever needed, and
which it used to derive from the raw export itself.

A partially malformed export still yields a usable dataset. A region missing
its label or its points is dropped with a warning rather than failing the read:
the alternative is one bad polygon in a batch of six hundred images costing a
training run, and the warning is what makes it findable afterwards.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import structlog

from digitex.domain.annotations import AnnotatedImage, LabelledRegion
from digitex.domain.entities import NormalizedPolygon
from digitex.domain.geometry import percent_to_normalized
from digitex.labeling.uris import local_file_path

if TYPE_CHECKING:
    from pathlib import Path

    from digitex.domain.entities import PercentPolygon

logger = structlog.get_logger()


def _open_ring(polygon: NormalizedPolygon) -> NormalizedPolygon:
    """*polygon* without a closing point, if it carries one.

    Label Studio stores an open ring, so a first point repeated as the last is
    not the tool's doing: it is a pre-annotation that reached the project already
    closed, kept by an annotator who had no reason to notice two handles stacked
    on one corner. 116 of the 3621 polygons in the training set are like that.
    The predictor no longer produces them, and a label file should not carry a
    vertex that says nothing about the region either way.
    """
    if len(polygon) > 1 and polygon[0] == polygon[-1]:
        return NormalizedPolygon(polygon[:-1])
    return polygon


def _regions(entry: dict[str, Any]) -> tuple[LabelledRegion, ...]:
    """Every usable region on one entry, in the order the export lists them."""
    usable: list[LabelledRegion] = []

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

        # An empty points list raises nothing on the way through, and would
        # reach the label file as a class id with no coordinates — an invalid
        # YOLO instance.
        if not normalized:
            logger.warning("skipped_polygon", reason="no points", polygon=polygon)
            continue

        usable.append(LabelledRegion(label=label, polygon=_open_ring(normalized)))

    return tuple(usable)


def read_export(path: Path) -> list[AnnotatedImage]:
    """Every annotated image in the export at *path*.

    Entries whose image URI names no local file are dropped with a warning — a
    task synced from blob storage has no page on this disk to train on.

    Duplicate filenames come back as they were found rather than being resolved
    here: the export addresses images by URI and only the basename survives, so
    two annotation batches can each hold a ``30.jpg``. What to do about that is
    for whoever is assembling a dataset out of them to decide.

    Args:
        path: The exported JSON file, in Label Studio's JSON-MIN shape.

    Returns:
        One :class:`~digitex.domain.annotations.AnnotatedImage` per readable
        entry, in export order.
    """
    with path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    images: list[AnnotatedImage] = []
    for entry in entries:
        uri = entry["image"]
        local = local_file_path(uri)
        if local is None:
            logger.warning("skipped_entry_no_local_path", image=uri)
            continue
        images.append(AnnotatedImage(filename=local.name, regions=_regions(entry)))

    logger.info("read_export", images=len(images), path=str(path))
    return images
