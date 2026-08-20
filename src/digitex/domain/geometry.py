"""Conversions between the polygon spaces.

Label Studio stores polygon points as percentages (0-100) of the image size;
YOLO label files use 0-1; a mask off a page is in source pixels. Each space is
its own type, and every hop between them is here.

Pure arithmetic, so it belongs in ``domain``: the spaces themselves are named
in :mod:`digitex.domain.entities` and nothing here imports beyond them. Parsing
the tool's own URIs is not arithmetic and lives with the tool, in
:mod:`digitex.labeling.uris`. Scaling a YOLO mask up to pixels belongs to
:mod:`digitex.ml.predictors`, which is where masks come from.
"""

from digitex.domain.entities import NormalizedPolygon, PercentPolygon, PixelPolygon


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
