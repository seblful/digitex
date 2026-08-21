"""Conversions between the polygon spaces.

A polygon passes through three coordinate systems on its way from a YOLO mask
to a crop or an annotation: source-image pixels, Label Studio's percentages of
the image size (0-100), and the 0-1 range YOLO label files are written in. Each
space is its own type, and every hop between them is one function here.

Pure arithmetic, which is why it sits in ``domain``: the spaces are named in
:mod:`digitex.domain.entities` and nothing here reaches past them. Two nearby
jobs deliberately live elsewhere — parsing the tool's own URIs is not
arithmetic and belongs with the tool (:mod:`digitex.labeling.uris`), and
scaling a YOLO mask up into pixels belongs where masks come from
(:mod:`digitex.ml.predictors`).
"""

from digitex.domain.entities import NormalizedPolygon, PercentPolygon, PixelPolygon


def percent_to_normalized(points: PercentPolygon) -> NormalizedPolygon:
    """Convert Label Studio percent points (0-100) to normalized (0-1)."""
    return NormalizedPolygon([(x / 100, y / 100) for x, y in points])


def percent_to_pixel(
    points: PercentPolygon, img_width: int, img_height: int
) -> PixelPolygon:
    """Convert Label Studio percent points (0-100) to pixels.

    The way back in for an annotation that has to be measured against the page
    it was drawn on, rather than trained from. Rounds to whole pixels because
    that is what :data:`~digitex.domain.entities.PixelPolygon` is, and a
    hundredth of a percent of a 2400 px page is a quarter of one.
    """
    return PixelPolygon(
        [(round(x / 100 * img_width), round(y / 100 * img_height)) for x, y in points]
    )


def pixel_to_percent(
    polygon: PixelPolygon, img_width: int, img_height: int
) -> PercentPolygon:
    """Convert pixel points to Label Studio percent points (0-100)."""
    return PercentPolygon(
        [[x / img_width * 100, y / img_height * 100] for x, y in polygon]
    )
