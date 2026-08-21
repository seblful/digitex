"""Snapping a hand-traced outline onto the print it was drawn around.

An annotator's polygon says *which* text belongs to a question. It does not say
where that text ends: a hand-traced outline carries whatever slack the mouse
left, and no two carry the same slack. Over the 3621 outlines of the page
corpus, the middle eighty percent sat anywhere from 0.14 to 0.45 line heights
from their own print, eighty pairs overlapped each other, and 126 cut through
print they were meant to contain.

This rebuilds each outline from the ink it already holds, so every region ends
up the same distance from its own text. Four properties hold, in the order they
constrain the work:

1. **Nothing changes hands.** Every blob of ink is assigned to the region that
   already held most of it, once, before any outline moves
   (:func:`_ownership`). A rebuilt outline is then built from its own ink alone,
   so it can neither be held out by a neighbour's descender nor reach across and
   take a neighbour's line.
2. **No print is dropped.** Every blob a region owned is inside the outline it
   gets back. Argued by construction and then *checked* outright, because the
   territory split can sever a claim in two and a polygon is one ring: a region
   the rebuild would strand keeps the outline it came in with.
3. **No two outlines overlap.** White space two regions both reach for goes to
   whichever one's print is nearer (:func:`_territory`), so the partition is a
   fact of the construction rather than something checked afterwards. The
   thinning is inward-only for the same reason — see :func:`thinned`.
4. **The annotator's shape survives.** An edge may take up slack inside the
   original outline freely, but may not travel more than *grow* line heights
   beyond it. A region traced around a figure comes back around that figure.

All of it happens in the page's own level frame. The scans are tilted by up to
five degrees, and a margin measured on the scan's axes is a different margin at
each end of a tilted line.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import cv2
import numpy as np

from digitex.domain.entities import PixelPolygon
from digitex.imaging.ink import read_ink, rotation, row_runs, turn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image

# Target clearance between an outline and its print, in line heights. A quarter
# of a line is about what the print leaves between two lines of itself, so a
# region ends up as far from its own edge as its lines are from each other.
MARGIN = 0.25

# A run of inked rows shorter than this share of a line height is a fragment --
# a detached accent, the dot of a colon, the tail of a bracket -- and is read as
# part of the nearest line rather than as a line of its own.
#
# Deliberately not a gap threshold. These books are set with 2-7 px of leading
# at 300 dpi, which is the same order as the gap over an accent, so no gap tells
# a line break from a diacritic. A run's height does: a line of print stands
# 25-35 px tall and an accent 4.
FRAGMENT = 0.5

# How far apart two lines' ends may be and still share one edge, in line
# heights. A block of justified text is ragged by a few pixels a line and every
# one of those would otherwise cost two handles. A weak knob in practice:
# sweeping it across its whole range moved the finished handle count only from
# 20.8 to 19.9.
SNAP = 0.3

# How far outside the annotator's own outline an edge may travel, in line
# heights. The shape is the part of the annotation carrying a judgement; the
# slack is not.
GROW = 0.5

# No outline leaves here with more handles than this, budget permitting. The
# same figure :mod:`digitex.ml.predictors` thins a predicted outline to, so a
# rebuilt outline and a predicted one reach the editor alike.
BUDGET = 20

# Thinning stops here: ``cut_out_image_by_polygon`` needs four points to raise a
# quad from.
MIN_RING_POINTS = 4

# A region holding less print than this is left alone -- there is nothing to
# measure a margin against. Thirteen of the corpus's outlines are like that, and
# every one is worth a human look rather than a guess.
MIN_INK = 40

# Used only when a page turns up with no measurable line of print at all.
_FALLBACK_LINE = 24

# How much clearance the territory split leaves along a seam, in pixels. Enough
# that rounding the two rings onto the pixel grid cannot put them back on the
# same line; small enough to be invisible.
_SEPARATION = 2.0

# The area a vertex must span to be worth a handle, as a fraction of the page
# diagonal, squared -- as in :mod:`digitex.ml.predictors`, and for the same
# reason: an absolute floor thins a downscaled page five times as hard as a
# full-resolution one.
_SMOOTH_AREA = 0.003


@dataclass(frozen=True)
class Outline:
    """One labelled region, in source-image pixels."""

    label: str
    polygon: PixelPolygon


@dataclass(frozen=True)
class Aligned:
    """What became of one outline.

    A region the rebuild cannot speak for comes back unchanged with *reason*
    filled in, rather than getting a guess. That is what makes the pass safe to
    run over finished work: the worst it can do to a region is nothing.
    """

    label: str
    polygon: PixelPolygon
    changed: bool
    reason: str = ""


def _as_array(polygon: PixelPolygon | Sequence[tuple[int, int]]) -> np.ndarray:
    return np.asarray(polygon, dtype=np.float64).reshape(-1, 2)


def _as_polygon(points: np.ndarray) -> PixelPolygon:
    return PixelPolygon([(int(x), int(y)) for x, y in points])


def _whole_pixels(points: np.ndarray) -> np.ndarray:
    """*points* on the pixel grid, with the duplicates rounding created dropped.

    A :data:`~digitex.domain.entities.PixelPolygon` is whole pixels, so the
    rounding has to happen before the checks rather than after them -- rounding
    is itself a move, and two vertices a third of a pixel apart become one. A
    repeated vertex is not a shape, and it makes every orientation test on it
    degenerate.
    """
    whole = np.round(points)
    keep = [
        index
        for index in range(len(whole))
        if index == 0 or not np.array_equal(whole[index], whole[index - 1])
    ]
    if len(keep) > 1 and np.array_equal(whole[keep[0]], whole[keep[-1]]):
        keep.pop()
    return whole[keep]


def _moved(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """*points* carried through a 2x3 affine matrix."""
    return points @ matrix[:, :2].T + matrix[:, 2]


def _filled(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """*points* filled in as a 0/255 stencil the size of the page."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    return mask


def _ownership(ink: np.ndarray, masks: list[np.ndarray]) -> list[np.ndarray]:
    """Each region's own print, decided once for the whole page.

    A blob goes to the region holding most of it, and only if that region holds
    half of it outright: a letter straddling two outlines belongs to neither, and
    print no outline covers belongs to the page. Deciding this before any outline
    moves is what property 1 in the module docstring rests on.
    """
    count, blobs, _, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    # OpenCV's stubs leave the label map's dtype open; it is a blob index, and
    # every use below is either a bincount or a lookup that needs it to be one.
    labels = blobs.astype(np.intp)
    areas = np.bincount(labels.ravel(), minlength=count)
    held = np.zeros((max(len(masks), 1), count), dtype=np.int64)
    for index, mask in enumerate(masks):
        held[index] = np.bincount(labels[mask > 0], minlength=count)

    best = held.argmax(axis=0)
    owner = np.where(held[best, np.arange(count)] * 2 >= areas, best, -1)
    owner[0] = -1  # blob 0 is the paper
    return [
        ((owner[labels] == index) & (ink > 0)).astype(np.uint8) * 255
        for index in range(len(masks))
    ]


def _line_height(own: list[np.ndarray]) -> int:
    """The page's median line height, measured inside its own regions.

    Per region and pooled, never off the whole page: a page-wide row profile
    welds the question column to the answer boxes beside it, and the run it
    reports is two lines at different heights rather than one line.
    """
    runs = [bottom - top for mask in own for top, bottom in row_runs(mask, min_run=2)]
    return int(np.median(runs)) if runs else _FALLBACK_LINE


def _bands(mask: np.ndarray, fragment: int) -> list[tuple[int, int, int, int]]:
    """*mask*'s print as lines: (top, bottom, left, right) for each one."""
    runs = row_runs(mask)
    if not runs:
        return []

    lines = [run for run in runs if run[1] - run[0] >= fragment]
    if not lines:
        lines = [(min(top for top, _ in runs), max(bottom for _, bottom in runs))]

    joined = [list(line) for line in lines]
    for top, bottom in runs:
        if bottom - top >= fragment:
            continue
        # A fragment above the first line or below the last can only go one way;
        # between two it joins whichever it sits closer to.
        nearest = min(
            range(len(joined)),
            key=lambda index: min(
                abs(joined[index][0] - bottom), abs(top - joined[index][1])
            ),
        )
        joined[nearest][0] = min(joined[nearest][0], top)
        joined[nearest][1] = max(joined[nearest][1], bottom)

    bands: list[tuple[int, int, int, int]] = []
    for top, bottom in joined:
        columns = np.flatnonzero(mask[top:bottom].any(axis=0))
        if columns.size:
            bands.append((top, bottom, int(columns[0]), int(columns[-1]) + 1))
    return bands


def _snapped(bands: list[tuple[int, int, int, int]], snap: int) -> list[list[int]]:
    """*bands* with near-agreeing ends pulled onto one shared edge.

    The pull is toward the region's own widest line rather than onto an absolute
    grid: each line's shortfall from that edge is rounded down to a multiple of
    *snap*, so the lines reaching furthest all land on one edge exactly and a
    line falling short by less than a step joins them. Rounding the shortfall
    *down* is what keeps this safe -- an edge only ever moves outward from the
    print, never across it.
    """
    if snap > 0:
        widest = max(band[3] for band in bands)
        narrowest = min(band[2] for band in bands)
        quantized = [
            [
                top,
                bottom,
                narrowest + (left - narrowest) // snap * snap,
                widest - (widest - right) // snap * snap,
            ]
            for top, bottom, left, right in bands
        ]
    else:
        quantized = [list(band) for band in bands]

    groups = [quantized[0]]
    for top, bottom, left, right in quantized[1:]:
        if left == groups[-1][2] and right == groups[-1][3]:
            groups[-1][1] = bottom
        else:
            groups.append([top, bottom, left, right])
    return groups


def _stacked(
    groups: list[list[int]], margin: int, shape: tuple[int, int]
) -> list[list[int]]:
    """*groups* padded by *margin* and made vertically contiguous.

    Padding two neighbouring lines pushes them into each other; the seam goes at
    the midpoint of the overlap, so neither line's clearance is the one that
    gives way and the stack still closes into a single ring.
    """
    height, width = shape
    padded = [
        [
            top - margin,
            bottom + margin,
            max(left - margin, 0),
            min(right + margin, width),
        ]
        for top, bottom, left, right in groups
    ]
    for above, below in pairwise(padded):
        if above[1] > below[0]:
            seam = (above[1] + below[0]) // 2
            above[1] = below[0] = seam
    padded[0][0] = max(padded[0][0], 0)
    padded[-1][1] = min(padded[-1][1], height)
    return padded


def _staircase(bands: list[list[int]]) -> np.ndarray:
    """The single ring around a vertical stack of rectangles.

    Down the right-hand side taking each step as it comes, then back up the left.
    Consecutive rectangles of equal width contribute no step, which is what keeps
    a plain paragraph at four points.
    """
    right_side: list[tuple[int, int]] = []
    left_side: list[tuple[int, int]] = []
    for index, (top, bottom, left, right) in enumerate(bands):
        if index == 0 or right != bands[index - 1][3]:
            right_side.append((right, top))
        right_side.append((right, bottom))
        if index == 0 or left != bands[index - 1][2]:
            left_side.append((left, top))
        left_side.append((left, bottom))

    ring: list[tuple[int, int]] = []
    for point in right_side + left_side[::-1]:
        if not ring or point != ring[-1]:
            ring.append(point)
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring.pop()
    return np.array(ring, dtype=np.float64)


def _spanned(before, point, after) -> float:
    """The signed area *point* spans with its two neighbours."""
    return (
        (point[0] - before[0]) * (after[1] - before[1])
        - (after[0] - before[0]) * (point[1] - before[1])
    ) / 2


def _side(first, second, point) -> float:
    """Which side of the line *first*-*second* the *point* falls on."""
    return (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (
        point[0] - first[0]
    )


def _crosses(a, b, c, d) -> bool:
    """Whether segment a-b properly crosses segment c-d."""
    first, second = _side(c, d, a), _side(c, d, b)
    third, fourth = _side(a, b, c), _side(a, b, d)
    return ((first > 0) != (second > 0)) and ((third > 0) != (fourth > 0))


def tangled(ring: np.ndarray) -> bool:
    """Whether *ring* crosses itself anywhere but at a shared endpoint.

    A ring that crosses itself is what makes ``fillPoly`` punch a hole in the
    crop it was meant to mask, so nothing leaves this module without being asked.

    Straddling has to mean strictly opposite sides, with a tolerance around zero.
    A point lying *on* an edge is a touch, not a crossing, and once a ring has
    been rotated out of the level frame the exact zeros that meant "collinear"
    have become noise a hair either side of it. Comparing against zero alone
    reports those as crossings and rejects perfectly good outlines.
    """
    count = len(ring)
    if count < 4:
        return False

    # All edge pairs at once. The double loop this replaces was the whole cost of
    # a corpus run: a mask-traced ring can carry 160 vertices, and 25000
    # orientation tests per region in the interpreter dominated everything else.
    points = np.asarray(ring, dtype=np.float64)
    along = np.roll(points, -1, axis=0) - points
    offset = points[None, :, :] - points[:, None, :]
    side = along[:, None, 0] * offset[:, :, 1] - along[:, None, 1] * offset[:, :, 0]

    tolerance = 1e-6 * max(1.0, float(np.abs(points).max()) ** 2)
    behind = np.roll(side, -1, axis=1)
    straddles = ((side > tolerance) & (behind < -tolerance)) | (
        (side < -tolerance) & (behind > tolerance)
    )

    index = np.arange(count)
    touching = (
        (index[:, None] == index[None, :])
        | ((index[:, None] + 1) % count == index[None, :])
        | ((index[None, :] + 1) % count == index[:, None])
    )
    return bool((straddles & straddles.T & ~touching).any())


def _covers_ink(before, point, after, own: np.ndarray) -> bool:
    """Whether the triangle *point* spans holds any of the region's own print.

    Checked in the triangle's own bounding box rather than page-wide: a thinning
    pass asks this a few hundred times per page and the triangles are a handful
    of pixels across.
    """
    triangle = np.array([before, point, after], dtype=np.float64)
    left, top = np.floor(triangle.min(axis=0)).astype(int)
    right, bottom = np.ceil(triangle.max(axis=0)).astype(int) + 1
    height, width = own.shape
    left, top = max(int(left), 0), max(int(top), 0)
    right, bottom = min(int(right), width), min(int(bottom), height)
    if right <= left or bottom <= top:
        return False
    window = own[top:bottom, left:right]
    if not window.any():
        return False
    stencil = np.zeros(window.shape, dtype=np.uint8)
    cv2.fillPoly(stencil, [np.round(triangle - [left, top]).astype(np.int32)], 255)
    return bool((stencil & window).any())


def _would_tangle(points: list, index: int) -> bool:
    """Whether dropping ``points[index]`` makes the ring cross itself.

    Dropping a vertex replaces two edges with one, so the only edge that can
    start a crossing is that new one -- which is why this costs a walk of the
    ring rather than a walk of every pair.
    """
    count = len(points)
    start, end = points[index - 1], points[(index + 1) % count]
    # Edges index-1 and index are the two the new segment replaces, so they are
    # gone. Edges index-2 and index+1 survive but each shares an endpoint with
    # the new segment, and a shared endpoint is not a crossing.
    skip = {(index - 2) % count, (index - 1) % count, index, (index + 1) % count}
    return any(
        _crosses(start, end, points[step], points[(step + 1) % count])
        for step in range(count)
        if step not in skip
    )


def thinned(
    ring: np.ndarray, own: np.ndarray, min_area: float, budget: int
) -> np.ndarray:
    """*ring* with its least telling vertices dropped, least telling first.

    Visvalingam-Whyatt ordering, as in :mod:`digitex.ml.predictors`: what a
    vertex costs is the area it spans with its neighbours, so a staircase tread
    goes before a corner carrying the shape.

    Three extra conditions, and they are what make the thinning safe to run
    *after* the territory split rather than before it. A vertex may only go if
    dropping it makes the ring smaller, if the triangle it gives up holds none of
    the region's own print, and if the ring does not fold as a result. Monotone
    shrinking is the whole argument: a ring that only ever loses area cannot
    reach into a neighbour it was disjoint from. Plain Visvalingam-Whyatt has no
    such property — cutting a reflex corner pushes the edge outward, which is
    exactly how a thinned outline lands on top of the question below it.

    The price is that the budget is a wish rather than a promise: a ring whose
    every remaining vertex is reflex, or hemmed in by print, stops where it is.
    """
    points = [tuple(point) for point in ring]
    if len(points) <= MIN_RING_POINTS:
        return ring

    whole = sum(
        _spanned((0.0, 0.0), points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )
    outward = 1.0 if whole >= 0 else -1.0

    def spans(index: int) -> float:
        return _spanned(
            points[index - 1], points[index], points[(index + 1) % len(points)]
        )

    areas = [spans(index) for index in range(len(points))]
    while len(points) > MIN_RING_POINTS:
        # A vertex spanning exactly no area is collinear with its neighbours, or
        # the tip of a zero-width spur the mask trace left behind. Removing it
        # cannot move the outline, so it is always allowed -- and it has to be,
        # or it survives into the output and turns every orientation test on it
        # into a coin flip once the ring is rotated back out of the level frame.
        #
        # The guards are asked of candidates cheapest-first and no further:
        # rasterising a triangle and walking a ring both cost far more than
        # comparing two areas, and all but the first few are never reached.
        candidates = sorted(
            (index for index in range(len(points)) if areas[index] * outward >= 0),
            key=lambda index: abs(areas[index]),
        )
        chosen: int | None = None
        for index in candidates:
            if abs(areas[index]) > min_area and len(points) <= budget:
                # The cheapest vertex left is already worth keeping, so every
                # dearer one is too.
                break
            neighbours = (points[index - 1], points[(index + 1) % len(points)])
            if _covers_ink(neighbours[0], points[index], neighbours[1], own):
                continue
            if _would_tangle(points, index):
                continue
            chosen = index
            break
        if chosen is None:
            break
        del points[chosen]
        del areas[chosen]
        areas[chosen - 1] = spans(chosen - 1)
        areas[chosen % len(points)] = spans(chosen % len(points))
    return np.array(points, dtype=np.float64)


def _territory(own: list[np.ndarray], claims: list[np.ndarray]) -> list[np.ndarray]:
    """*claims* with every contested pixel given to the nearest print.

    Two regions reaching for the same white gap is the one way this rebuild can
    produce an overlap, and it is settled the way a reader settles it: the gap
    belongs to whichever block of print is closer. Own ink is at distance zero
    from itself, so no region can lose print to the split.
    """
    # One claim cannot contest anything, and none cannot be stacked at all --
    # a page where every region turned out to hold no print reaches here empty.
    if len(claims) < 2:
        return claims
    return _pushed_apart(own, _by_nearest_print(own, claims))


def _by_nearest_print(
    own: list[np.ndarray], claims: list[np.ndarray]
) -> list[np.ndarray]:
    """*claims* with every pixel two of them want given to the nearer print."""
    stacked = np.stack([(claim > 0).astype(np.uint8) for claim in claims])
    contested = stacked.sum(axis=0) > 1
    if not contested.any():
        return claims

    rows, columns = np.nonzero(contested)
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    distances = np.stack(
        [
            cv2.distanceTransform(
                255 - (mask[top:bottom, left:right] > 0).astype(np.uint8) * 255,
                cv2.DIST_L2,
                3,
            )
            for mask in own
        ]
    )
    # A region with no print of its own must not win a gap by default, and an
    # empty distance transform is uniform -- push it out of the running.
    for index, mask in enumerate(own):
        if not mask.any():
            distances[index] = np.inf
    nearest = distances.argmin(axis=0)

    window = contested[top:bottom, left:right]
    settled = []
    for index, claim in enumerate(claims):
        kept = claim.copy()
        kept[top:bottom, left:right][window & (nearest != index)] = 0
        settled.append(kept)
    return settled


def _pushed_apart(own: list[np.ndarray], claims: list[np.ndarray]) -> list[np.ndarray]:
    """*claims* with a hair of paper left between each one and its neighbours.

    Winning the split outright is not enough, and this is the subtler half of
    keeping outlines disjoint. Two claims can end up merely *touching* -- never
    contesting a single pixel, so the nearest-print split never looks at them --
    and a shared boundary line survives the rotation back onto the scan's own
    axes only to be rounded onto the pixel grid, both regions landing on it and
    both filling it. That is an overlap a hair wide and a few hundred pixels
    long, and it is invisible in the level frame where the split was decided.

    Own print is never given up to the gap: two regions' letters can sit a pixel
    apart, and there this would be eating print rather than paper.
    """
    if len(claims) < 2:
        return claims

    reach = int(_SEPARATION)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * reach + 1, 2 * reach + 1))
    presence = [(claim > 0).astype(np.uint8) for claim in claims]
    total = np.sum(presence, axis=0, dtype=np.int16)

    apart = []
    for index, claim in enumerate(claims):
        # Every other claim at once, rather than a pass per pair.
        others = ((total - presence[index]) > 0).astype(np.uint8) * 255
        crowded = cv2.dilate(others, kernel) > 0
        kept = claim.copy()
        kept[crowded & (own[index] == 0)] = 0
        apart.append(kept)
    return apart


def _ring_of(mask: np.ndarray) -> np.ndarray | None:
    """The outer ring of *mask*'s largest piece, holes closed over."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None
    return largest.reshape(-1, 2).astype(np.float64)


def _untangled(
    ring: np.ndarray | None, own: np.ndarray, min_area: float, budget: int
) -> np.ndarray | None:
    """The tightest thinning of *ring* that does not cross itself.

    Thinning to a budget is the last thing that can fold a ring, and when it
    does, the answer is to thin *less* rather than to give up on the region. A
    ring keeping sixty handles is still snapped to the print, still inside its
    own territory and still holding all its own ink; the annotator's original
    outline is none of those things.
    """
    if ring is None:
        return None
    for attempt in (
        thinned(ring, own, min_area, budget),
        thinned(ring, own, min_area, len(ring) + 1),
        ring,
    ):
        if len(attempt) >= MIN_RING_POINTS and not tangled(attempt):
            return attempt
    return None


def align_outlines(
    image: Image.Image,
    outlines: Sequence[Outline],
    *,
    margin: float = MARGIN,
    snap: float = SNAP,
    grow: float = GROW,
    budget: int = BUDGET,
) -> list[Aligned]:
    """Rebuild every outline on one page from the print it contains.

    Args:
        image: The page the outlines were drawn on.
        outlines: The outlines as the annotator left them, in source pixels.
        margin: Target clearance, in line heights.
        snap: How far apart two lines' ends may be and still share one edge, in
            line heights. Zero gives every line its own edge.
        grow: How far outside its original an outline may travel, in line
            heights.
        budget: Vertex budget for the thinning.

    Returns:
        One :class:`Aligned` per outline, in the order given. A region the
        rebuild cannot speak for comes back unchanged, carrying the reason.
    """
    if not outlines:
        return []

    page = read_ink(image)
    shape = page.shape
    height, width = shape
    to_level = rotation(shape, page.skew)
    to_source = cv2.invertAffineTransform(to_level)

    ink = turn(page.mask, page.skew) if page.skew else page.mask
    level = [_moved(_as_array(outline.polygon), to_level) for outline in outlines]
    masks = [_filled(points, shape) for points in level]
    own = _ownership(ink, masks)

    line = _line_height(own)
    pad = max(round(margin * line), 1)
    fragment = max(round(FRAGMENT * line), 2)
    step = max(round(snap * line), 1) if snap > 0 else 0
    reach = max(round(grow * line), 1)

    claims, refusals = _claims(
        own, masks, shape, pad=pad, fragment=fragment, step=step, reach=reach
    )

    # Paired up rather than indexed twice, so the Nones are gone from the type as
    # well as from the list.
    live = [(index, claim) for index, claim in enumerate(claims) if claim is not None]
    settled = _territory(
        [own[index] for index, _ in live], [claim for _, claim in live]
    )
    for (index, _), claim in zip(live, settled, strict=True):
        claims[index] = claim

    floor = (_SMOOTH_AREA * float(np.hypot(height, width))) ** 2
    return [
        _verdict(
            outline,
            claims[index],
            own[index],
            shape,
            to_level=to_level,
            to_source=to_source,
            floor=floor,
            budget=budget,
            refusal=refusals[index],
        )
        for index, outline in enumerate(outlines)
    ]


def _claims(
    own: list[np.ndarray],
    masks: list[np.ndarray],
    shape: tuple[int, int],
    *,
    pad: int,
    fragment: int,
    step: int,
    reach: int,
) -> tuple[list[np.ndarray | None], list[str]]:
    """What each region asks for, before any of them is told what it may have.

    One claim per region, built from that region's own print alone and clipped to
    the licence its original outline gives it. Claims may still overlap each
    other on the way out of here; :func:`_territory` is what settles that.
    """
    claims: list[np.ndarray | None] = []
    refusals: list[str] = []
    for index, mask in enumerate(masks):
        if int((own[index] > 0).sum()) < MIN_INK:
            claims.append(None)
            refusals.append("no print of its own")
            continue
        bands = _bands(own[index], fragment)
        built = _filled(_staircase(_stacked(_snapped(bands, step), pad, shape)), shape)
        # The annotator's own outline is the licence: an edge may take up the
        # slack inside it freely and step *grow* line heights past it, no
        # further. Measured as a distance rather than dilated with a kernel that
        # wide -- the kernel is the same answer at forty times the cost.
        outside = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        capped = cv2.bitwise_and(built, ((outside <= reach) * 255).astype(np.uint8))
        # Keeping the region's print outranks holding it near the annotator's
        # line, so if the cap would cut print, the cap is what gives way. OR-ing
        # the ink back in instead would leave the outline tracing letters.
        lost = bool(((own[index] > 0) & (capped == 0)).any())
        claims.append(built if lost else capped)
        refusals.append("")
    return claims, refusals


def _verdict(
    outline: Outline,
    claim: np.ndarray | None,
    own: np.ndarray,
    shape: tuple[int, int],
    *,
    to_level: np.ndarray,
    to_source: np.ndarray,
    floor: float,
    budget: int,
    refusal: str,
) -> Aligned:
    """One region's settled claim as a ring, or the outline it came in with.

    Where the promises this module makes are kept or given up on. A region gets
    its rebuild only if the finished ring holds every pixel of its own print and
    does not cross itself; otherwise the annotator's outline stands and says why.
    """
    height, width = shape
    ring = _untangled(
        _ring_of(claim) if claim is not None else None, own, floor, budget
    )

    source: np.ndarray | None = None
    if ring is not None:
        # Rotating the level-frame ring back can put a corner a few pixels past
        # the edge of the scan -- the frame was levelled on a canvas its own
        # size. At under fifteen pixels a clamp costs the shape nothing, and it
        # happens before the checks below because the clamp is itself a move that
        # can fold a ring: what gets checked has to be what gets returned.
        source = _moved(ring, to_source)
        np.clip(source[:, 0], 0, width, out=source[:, 0])
        np.clip(source[:, 1], 0, height, out=source[:, 1])
        source = _whole_pixels(source)
        if len(source) < MIN_RING_POINTS:
            source = None

    reason = refusal or ("no usable ring" if source is None else "")
    if source is not None and not reason:
        # The guarantee, checked rather than argued. Every step is meant to hold
        # the region's print, but a claim severed in two by the territory split
        # leaves a ring around only the larger piece -- and a polygon is one
        # ring, so the smaller piece cannot come along.
        held = _filled(_moved(source, to_level), shape)
        if ((own > 0) & (held == 0)).any():
            reason = "rebuild would strand some of its print"
        elif tangled(source):
            reason = "rebuild would come out self-crossing"

    if reason:
        return Aligned(outline.label, outline.polygon, False, reason)
    assert source is not None
    return Aligned(outline.label, _as_polygon(source), True)
