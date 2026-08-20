"""Reading a scan's black and white points, and correcting to them.

No scan is black on white. The lamp falls off toward the spine, the paper is
gray, and two pages of the same book are not the same gray. So nothing here
trusts a fixed threshold — the page's own histogram is asked where its ink and
its paper actually sit, and only what lies between the two is stretched.

Three passes, in the order :mod:`digitex.imaging.document` runs them:
:func:`flatten_scan` divides the uneven light out, :func:`scan_levels` finds
the two points, :func:`whiten_scan` applies them. Each reads statistics the
pass before it changed, which is why they stay three calls and not one.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

# The square each background sample summarises. Wider than any run of ink —
# a tile always catches some paper between text lines — and narrower than
# the shadows it must follow.
_BG_TILE = 128

# The percentile of a tile that reads as its paper: high enough to look past
# the ink, low enough to duck under the odd blown-out pixel.
_BG_PERCENTILE = 90

# The percentile of the background grid taken as the true paper level — the
# brightness every patch is corrected toward.
_BG_TARGET_PERCENTILE = 95

# The most a patch is brightened. Enough for the deepest shadow in the
# archive; the cap is what keeps a dark region too large for the grid's
# median to rescue from being blown out toward white.
_BG_MAX_GAIN = 4.0


# The percentile of a peak's mass at which its white/black point is set. Zero
# flattens all of the near-white and near-black to pure, one keeps the noise.
_PERCENTILE = 0.2

# Correcting in a luminance space needs more granularity than 0-255 gives.
_GAMMA = 2.2

_LUM_MULTIPLIER = 16

_MAX_LUM = 255 * _LUM_MULTIPLIER

# Stored level to luminance. The dark end is spread out and the multiplier
# gives it room, so a stretch from a black point well above zero has somewhere
# to put the midtones instead of shifting them.
_LUMINANCE = tuple(
    round((value / 255) ** (1 / _GAMMA) * _MAX_LUM) for value in range(256)
)


def _background_grid(gray: np.ndarray) -> np.ndarray:
    """The paper's own brightness, one sample per tile.

    A bright percentile per tile rather than a mean, so the ink a tile holds
    does not read as darkness. A tile holding no paper at all — a figure's
    interior, a filled table cell — reads dark regardless and would be
    brightened into a smear, so the grid is median-filtered to hand such a
    tile its neighbours' paper instead, then blurred so that no tile boundary
    shows through in the output.
    """
    height, width = gray.shape
    # Padded up to whole tiles so the reshape below can split the page evenly.
    # Edge padding, so the part-tile at the margin reads as the paper beside it.
    padded = np.pad(
        gray, ((0, -height % _BG_TILE), (0, -width % _BG_TILE)), mode="edge"
    )
    tiles = padded.reshape(
        padded.shape[0] // _BG_TILE, _BG_TILE, padded.shape[1] // _BG_TILE, _BG_TILE
    )
    grid = np.percentile(tiles, _BG_PERCENTILE, axis=(1, 3)).astype(np.float32)
    # Both kernels are 3 wide, so a page under three tiles either way has
    # nothing to filter and keeps its raw grid.
    if min(grid.shape) >= 3:
        grid = cv2.GaussianBlur(cv2.medianBlur(grid, 3), (3, 3), 0)
    return grid


def flatten_scan(image: Image.Image) -> Image.Image:
    """Lift a scan's shadows by dividing its illumination out.

    A page is lit unevenly — a gutter shadow near the spine, a sag where the
    paper lifted off the glass — and no global correction fixes that:
    stretching levels turns a shadow darker rather than whiter, and hands the
    black point to the shadow instead of to the ink. Illumination is
    multiplicative, so the answer is a division: estimate the paper's
    brightness everywhere, then scale each patch by what it is missing. The
    same division hands ink inside a shadow its unshadowed tone back.

    Runs before :func:`whiten_scan`: with the paper flat, the histogram's
    white peak is tight and the black point falls to the ink where it belongs.

    Args:
        image: Input page, color or grayscale.

    Returns:
        Grayscale ("L") page, same size as the input.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    height, width = gray.shape
    grid = _background_grid(gray)

    rows, columns = grid.shape
    background = cv2.resize(
        grid, (columns * _BG_TILE, rows * _BG_TILE), interpolation=cv2.INTER_LINEAR
    )[:height, :width]
    paper = float(np.percentile(grid, _BG_TARGET_PERCENTILE))

    # Clipped at 1.0 so the gain only ever brightens: anything already above
    # the paper level is the scanner's saturated canvas, and pulling that down
    # would draw a gray border round the page. The floor of 1.0 under the
    # background is for a tile that came back pure black.
    gain = np.clip(paper / np.maximum(background, 1.0), 1.0, _BG_MAX_GAIN)
    return Image.fromarray(np.clip(gray * gain, 0, 255).astype(np.uint8), mode="L")


@dataclass(frozen=True)
class _Peak:
    """A histogram peak and the basin it drains, as shares of all pixels."""

    value: int
    height: float
    left: int
    right: int
    left_bottom: float
    right_bottom: float
    mass: float


def _share(shares: list[float], level: int) -> float:
    """A level's share of the page, reading past either end of the range as 0."""
    return shares[level] if 0 <= level < 256 else 0.0


def _basin(
    shares: list[float], value: int, mass: float, step: int
) -> tuple[int, float, float]:
    """One side of the basin around level *value*, walked *step* at a time.

    Descends while the histogram keeps falling, tolerating a single level that
    rises — a one-bin notch is noise inside a peak, not the next peak along.

    *mass* arrives as the share collected so far and leaves with this side's
    added, rather than each side returning its own subtotal. The two sides are
    chained through one accumulator on purpose: a peak's mass decides which
    peak is the paper, and re-associating the additions would move it by an
    ulp — enough, at a tie, to pick the other peak.

    Returns:
        How far the basin reaches, how low it gets, and the running mass.
    """
    edge, floor = value, shares[value]
    level = value + step
    while 0 <= level < 256:
        if shares[level] >= floor and _share(shares, level + step) >= floor:
            break
        floor = min(floor, shares[level])
        edge = level
        mass += shares[level]
        level += step
    return edge, floor, mass


def _find_peaks(shares: list[float]) -> list[_Peak]:
    """Every level that outranks its two neighbours on each side, with basin."""
    peaks = []
    for value in range(256):
        here = shares[value]
        if not all(here > _share(shares, value + d) for d in (-2, -1, 1, 2)):
            continue
        left, left_bottom, mass = _basin(shares, value, here, -1)
        right, right_bottom, mass = _basin(shares, value, mass, 1)
        peaks.append(_Peak(value, here, left, right, left_bottom, right_bottom, mass))
    return peaks


def _magnitude(height: float) -> float:
    return math.log10(1e4 * height + 1)


def _white_peak_score(peak: _Peak) -> float:
    """Favours a bright peak carrying a lot of the page — the paper."""
    # The linear tail below a tenth of the page keeps the curve continuous
    # where the two halves meet, and keeps log10 away from an empty peak.
    mass = math.log10(100 * peak.mass) if peak.mass > 0.1 else 10 * peak.mass
    return (peak.value / 255) ** 3 * mass


def _black_peak_score(peak: _Peak) -> float:
    """Favours a dark peak standing well clear of its basin — the ink."""
    relief = _magnitude(peak.height) - _magnitude(
        min(peak.left_bottom, peak.right_bottom)
    )
    return (1 - peak.value / 255) ** 3 * relief


def _shoulder(
    counts: list[int], peak: _Peak, *, from_below: bool, percentile: float
) -> int:
    """The level a *percentile* of the way into *peak* from one side.

    Neither the peak's edge nor its crest. Clipping a share of the mass on
    purpose is what flattens near-white to white and near-black to black
    rather than leaving both as noise. The share is measured against every
    level from the basin edge outward, not against the peak alone.

    Falls back to the far side of the peak, for a histogram whose mass sits
    entirely outside the basin the walk covers.
    """
    if from_below:
        levels = range(peak.left, peak.right)
        tail = sum(counts[peak.left :])
        fallback = peak.right
    else:
        levels = range(peak.right, peak.left, -1)
        tail = sum(counts[: peak.right + 1])
        fallback = peak.left

    seen = 0
    for level in levels:
        seen += counts[level]
        if seen >= percentile * tail:
            return level
    return fallback


def _levels_from(
    counts: list[int], white_percentile: float = _PERCENTILE
) -> tuple[tuple[int, int] | None, int]:
    """The (black, white) points *counts* implies, and the white peak's level.

    The level comes back so the caller can tell a paper peak from the scan's
    saturated margin; it is -1 when there was no peak to pick.
    """
    total = sum(counts)
    if not total:
        return None, -1

    peaks = _find_peaks([count / total for count in counts])
    if not peaks:
        return None, -1

    white_peak = max(peaks, key=_white_peak_score)
    black_peak = max(peaks, key=_black_peak_score)
    if white_peak.value <= black_peak.value:
        return None, white_peak.value

    black = _shoulder(counts, black_peak, from_below=False, percentile=_PERCENTILE)
    white = _shoulder(counts, white_peak, from_below=True, percentile=white_percentile)
    # The two basins can overlap even when their peaks do not, and a ramp
    # needs somewhere to run.
    levels = (black, white) if black < white else None
    return levels, white_peak.value


def content_box(pixels: np.ndarray) -> tuple[slice, slice]:
    """Where the page itself sits, with the scanner's white canvas cut off.

    A line belongs to the page when more than half of it is not pure white.
    On a scan the two are far apart — paper is grey and the canvas is
    saturated — so the edge is sharp and needs no tolerance.
    """
    height, width = pixels.shape
    inked = pixels != 255
    rows = np.flatnonzero(inked.mean(axis=1) > 0.5)
    columns = np.flatnonzero(inked.mean(axis=0) > 0.5)
    # A blank scan, or one whose page is a thin stripe: nothing found in one
    # direction means there is no box to trust, so the caller keeps the lot.
    if not rows.size or not columns.size:
        return slice(0, height), slice(0, width)
    return (
        slice(int(rows[0]), int(rows[-1]) + 1),
        slice(int(columns[0]), int(columns[-1]) + 1),
    )


def _page_counts(gray: Image.Image) -> list[int]:
    """The histogram of the page alone: no scan margin, no clipped level."""
    pixels = np.array(gray)
    page = pixels[content_box(pixels)]
    counts = np.bincount(page.ravel(), minlength=256).tolist()
    counts[255] = 0
    return counts


def scan_levels(
    image: Image.Image, white_percentile: float = _PERCENTILE
) -> tuple[int, int] | None:
    """The (black, white) points to correct *image* against.

    It reads the page's histogram, picks the peak that looks like ink and the
    peak that looks like paper, then sets each point a fifth of the way into
    its peak — inside the noise rather than at its edge, so near-black and
    near-white flatten out. That much is NAPS2's.

    Where this parts company with NAPS2 is the second look. A scanner lays
    the page on a pure-white canvas, and where the margin is wide it can
    outvote the paper — NAPS2 then corrects the page against its own border
    and leaves the paper gray. Pure white winning the contest is the tell, so
    on those pages the margin is cropped away and the level everything blown
    out piles up in is dropped, which leaves the paper as the brightest thing
    left to find.

    Args:
        image: Input page, color or grayscale.
        white_percentile: How far into the paper peak the white point sits.
            The default is NAPS2's fifth, sized for a raw scan's wide peak; a
            flattened page's peak is tight, and the same fifth lands so close
            under the paper that a gray film survives the stretch.

    Returns:
        The two levels, or None when the histogram gives no usable pair and
        the page is better left alone.
    """
    gray = image.convert("L")
    levels, white_peak = _levels_from(gray.histogram(), white_percentile)
    if white_peak == 255:
        levels, _ = _levels_from(_page_counts(gray), white_percentile)
    return levels


def _correction_ramp(black: int, white: int) -> list[int]:
    """The 256-entry curve that clamps to [*black*, *white*] and stretches."""
    black_lum, white_lum = _LUMINANCE[black], _LUMINANCE[white]

    ramp = []
    for value in range(256):
        # Clamped first, so everything past a point lands exactly on pure
        # rather than running off the end of the stretch.
        lum = _LUMINANCE[min(max(value, black), white)]
        scaled = (lum - black_lum) * _MAX_LUM // (white_lum - black_lum)
        ramp.append(round((scaled / _MAX_LUM) ** _GAMMA * 255))
    return ramp


def whiten_scan(
    image: Image.Image, levels: tuple[int, int] | None = None
) -> Image.Image:
    """Burn a scan's gray paper out to white, deepening the ink to match.

    Clamps to the page's black and white points and stretches what is left
    across the full range. The stretch happens in a luminance space rather
    than straight on the stored values, which is what keeps the midtones from
    shifting when the black point is well above zero.

    Not a threshold: text keeps its anti-aliased edges, which is what the
    detector and OCR were trained on.

    Args:
        image: Input page, color or grayscale.
        levels: The (black, white) points to correct against. Read off the
            page itself when omitted; a page whose histogram offers no usable
            pair is returned as-is, the way NAPS2 declines to correct it.

    Returns:
        Grayscale ("L") page, same size as the input.
    """
    gray = image.convert("L")
    points = scan_levels(gray) if levels is None else levels
    if points is None:
        return gray
    return gray.point(_correction_ramp(*points))
