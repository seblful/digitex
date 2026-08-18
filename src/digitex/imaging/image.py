"""Image processing utilities."""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from digitex.domain.entities import PixelPolygon


def resize_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    return ImageOps.contain(
        image, (max_width, max_height), method=Image.Resampling.BILINEAR
    )


# --- background flatten ---


def add_white_background(image: Image.Image) -> Image.Image:
    """Composite an image onto a white background.

    A crop's polygon mask leaves everything outside the region transparent, and
    JPEG has no alpha channel — the transparency must be flattened onto white
    before saving.

    Args:
        image: Input PIL Image.

    Returns:
        RGB image suitable for JPG format.
    """
    img = np.array(image.convert("RGBA"))
    alpha = img[:, :, 3:4] / 255.0
    white_bg = np.ones_like(img[:, :, :3]) * 255
    rgb = img[:, :, :3] * alpha + white_bg * (1 - alpha)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


# --- scan cleanup ---
#
# A port of NAPS2's document correction, the manual pass the book archive was
# scanned with: WhiteBlackPointOp to fix the paper's calibration, then
# BilateralFilterOp to take the grain off. Constants and formulas follow
# NAPS2.Images/Bitwise/ so the two stay comparable.
#
# One pass runs ahead of NAPS2's two and is not theirs: an illumination
# flatten (:func:`flatten_scan`), because the archive's gutter shadows are
# spatial and no global correction can touch them. Its constants were
# calibrated against the archive itself — the 2023 books carry the deepest
# shadows, needing a gain of about 3.2.

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

# The white shoulder for a flattened page. Flattening pulls the paper into a
# peak so tight that NAPS2's fifth (_PERCENTILE) lands barely under it and a
# gray film survives the stretch; a fiftieth reaches under the whole peak.
# Calibrated against the archive's figure pages — halftone tones survive it.
_FLAT_WHITE_PERCENTILE = 0.02

# Everything below this is ink or figure, not paper — the mass the black
# point is set from.
_INK_CEILING = 128

# The black point sits this far into the page's dark mass, as a percentile.
# At 300 dpi a stroke is mostly anti-aliasing — its core peak holds little of
# its mass — so the peak shoulder alone leaves text gray. A tenth of the way
# in is enough to turn the cores solid while the halo keeps its ramp;
# calibrated against the archive's halftone figures, which lose visible
# tone by a quarter of the way in.
_FLAT_BLACK_PERCENTILE = 10

# The percentile of a peak's mass at which its white/black point is set. Zero
# flattens all of the near-white and near-black to pure, one keeps the noise.
_PERCENTILE = 0.2

# Correcting in a luminance space needs more granularity than 0-255 gives.
_GAMMA = 2.2
_LUM_MULTIPLIER = 16
_MAX_LUM = 255 * _LUM_MULTIPLIER

# The filter's square around each pixel, and the gray distance at which a
# neighbour's weight reaches zero.
_FILTER_SIZE = 15
_FILTER_HALF = _FILTER_SIZE // 2
_COLOR_DIST_MAX = 32

# Vectorising the filter means sweeping the image once per neighbour, so a
# whole page would cross the memory bus 225 times. Working a tile at a time
# keeps the sweep inside cache instead. Purely a matter of speed — the output
# does not depend on it.
_TILE_SIZE = 256


def flatten_scan(image: Image.Image) -> Image.Image:
    """Lift a scan's shadows by dividing its illumination out.

    A page is lit unevenly — a gutter shadow near the spine, a sag where the
    paper lifted off the glass — and a global correction cannot fix that:
    stretching levels turns a shadow darker rather than whiter, and hands
    the black point to the shadow instead of the ink. Illumination is
    multiplicative, so the fix is a division: estimate the paper's
    brightness everywhere and scale each pixel by what its patch is missing.
    The same division hands ink inside a shadow its unshadowed tone back.

    The estimate is a coarse grid — a bright percentile per tile, so ink
    does not read as darkness. A tile holding no paper at all (a figure's
    interior, a filled table cell) reads dark all the same and would be
    brightened into a smear, so the grid is median-filtered to hand it its
    neighbours' paper instead, then blurred so no tile boundary shows in the
    output. Anything brighter than the paper — the scanner's saturated
    canvas — is left alone rather than darkened.

    Runs before :func:`whiten_scan`: with the paper flat, the histogram's
    white peak is tight and the black point falls to the ink where it
    belongs.

    Args:
        image: Input page, color or grayscale.

    Returns:
        Grayscale ("L") page, same size as the input.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    height, width = gray.shape
    padded = np.pad(
        gray, ((0, -height % _BG_TILE), (0, -width % _BG_TILE)), mode="edge"
    )
    tiles = padded.reshape(
        padded.shape[0] // _BG_TILE, _BG_TILE, padded.shape[1] // _BG_TILE, _BG_TILE
    )
    grid = np.percentile(tiles, _BG_PERCENTILE, axis=(1, 3)).astype(np.float32)
    if min(grid.shape) >= 3:
        grid = cv2.GaussianBlur(cv2.medianBlur(grid, 3), (3, 3), 0)
    background = cv2.resize(
        grid, (padded.shape[1], padded.shape[0]), interpolation=cv2.INTER_LINEAR
    )[:height, :width]
    paper = float(np.percentile(grid, _BG_TARGET_PERCENTILE))
    gain = np.clip(paper / np.maximum(background, 1.0), 1.0, _BG_MAX_GAIN)
    return Image.fromarray(np.clip(gray * gain, 0, 255).astype(np.uint8), mode="L")


@dataclass
class _Peak:
    """A histogram peak and the basin it drains, as shares of all pixels."""

    value: int
    height: float
    left: int
    right: int
    left_bottom: float
    right_bottom: float
    mass: float


def _find_peaks(shares: list[float]) -> list[_Peak]:
    """Every level that outranks its two neighbours on each side, with basin."""

    def share(level: int) -> float:
        return shares[level] if 0 <= level < 256 else 0.0

    peaks = []
    for value in range(256):
        here = shares[value]
        if not all(here > share(value + d) for d in (-2, -1, 1, 2)):
            continue

        peak = _Peak(value, here, value, value, here, here, here)
        for level in range(value - 1, -1, -1):
            if not (
                shares[level] < peak.left_bottom or share(level - 1) < peak.left_bottom
            ):
                break
            peak.left_bottom = min(peak.left_bottom, shares[level])
            peak.left = level
            peak.mass += shares[level]
        for level in range(value + 1, 256):
            if not (
                shares[level] < peak.right_bottom
                or share(level + 1) < peak.right_bottom
            ):
                break
            peak.right_bottom = min(peak.right_bottom, shares[level])
            peak.right = level
            peak.mass += shares[level]
        peaks.append(peak)
    return peaks


def _magnitude(height: float) -> float:
    return math.log10(1e4 * height + 1)


def _white_peak_score(peak: _Peak) -> float:
    """Favours a bright peak carrying a lot of the page — the paper."""
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
    """Walk into *peak* until *percentile* of its mass is behind us."""
    if from_below:
        levels = range(peak.left, peak.right)
        total = sum(counts[peak.left :])
        fallback = peak.right
    else:
        levels = range(peak.right, peak.left, -1)
        total = sum(counts[: peak.right + 1])
        fallback = peak.left

    seen = 0
    for level in levels:
        seen += counts[level]
        if seen >= percentile * total:
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


def _content_box(pixels: np.ndarray) -> tuple[slice, slice]:
    """Where the page itself sits, with the scanner's white canvas cut off.

    A line belongs to the page when more than half of it is not pure white.
    On a scan the two are far apart — paper is grey and the canvas is
    saturated — so the edge is sharp and needs no tolerance.
    """
    height, width = pixels.shape
    inked = pixels != 255
    rows = np.flatnonzero(inked.mean(axis=1) > 0.5)
    columns = np.flatnonzero(inked.mean(axis=0) > 0.5)
    if not rows.size or not columns.size:
        return slice(0, height), slice(0, width)
    return (
        slice(int(rows[0]), int(rows[-1]) + 1),
        slice(int(columns[0]), int(columns[-1]) + 1),
    )


def _page_counts(gray: Image.Image) -> list[int]:
    """The histogram of the page alone: no scan margin, no clipped level."""
    pixels = np.array(gray)
    page = pixels[_content_box(pixels)]
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
    to_lum = [round((value / 255) ** (1 / _GAMMA) * _MAX_LUM) for value in range(256)]
    black_lum, white_lum = to_lum[black], to_lum[white]

    ramp = []
    for value in range(256):
        lum = to_lum[min(max(value, black), white)]
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


def _spatial_weights() -> np.ndarray:
    """Neighbour weight by distance: full at the centre, nothing at a corner."""
    offsets = np.arange(_FILTER_SIZE) - _FILTER_HALF
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    furthest = math.sqrt(2 * _FILTER_HALF**2)
    return ((1 - np.hypot(dx, dy) / furthest) * 256).astype(np.int32)


def _filter_block(
    padded: np.ndarray,
    block: np.ndarray,
    top: int,
    left: int,
    spatial: np.ndarray,
) -> np.ndarray:
    """Weigh every pixel of *block* against its square of neighbours.

    *padded* holds the whole page with a half-window margin, so *top* and
    *left* — the block's origin on the page — index straight into it.
    """
    height, width = block.shape
    centre = block.astype(np.int16)
    weights = np.zeros(block.shape, dtype=np.int32)
    totals = np.zeros(block.shape, dtype=np.int32)

    for row in range(_FILTER_SIZE):
        for column in range(_FILTER_SIZE):
            nearness = int(spatial[row, column])
            if not nearness:
                continue
            neighbour = padded[
                top + row : top + row + height, left + column : left + column + width
            ]
            # NAPS2's weight table, worked out rather than looked up: a
            # neighbour counts for less the further off its tone is, and for
            # nothing at all past _COLOR_DIST_MAX. Arithmetic beats a lookup
            # here because the table is too big to stay in cache.
            tone = np.abs(centre - neighbour)
            np.subtract(_COLOR_DIST_MAX, tone, out=tone)
            np.maximum(tone, 0, out=tone)

            weight = tone.astype(np.int32) * nearness
            weights += weight
            totals += weight * neighbour

    return (totals // weights).astype(np.uint8)


def denoise_scan(image: Image.Image) -> Image.Image:
    """Average scanner grain out of a page without softening the text.

    A bilateral filter weighs a neighbour by how far away it is *and* how
    different it is, so grain within the paper and within a stroke averages
    away while the edge between them survives.

    Runs after :func:`whiten_scan`, for the reason NAPS2 gives: the filter
    reads distance in raw gray levels, so a page whose range is still
    compressed loses fine detail to it — and the pass is cheaper once the
    paper is uniform, which is what the untouched runs of white buy.

    Args:
        image: Input page, color or grayscale.

    Returns:
        Grayscale ("L") page, same size as the input.
    """
    gray = np.array(image.convert("L"))
    height, width = gray.shape
    padded = np.pad(gray, _FILTER_HALF, mode="edge")
    spatial = _spatial_weights()

    # Pixels within the filter's reach of an edge are left alone, as is any
    # white pixel between two others — an untouched run of paper.
    edge = _FILTER_HALF + 1
    inside = np.zeros(gray.shape, dtype=bool)
    inside[edge : height - _FILTER_HALF, edge : width - _FILTER_HALF] = True
    white = gray == 255
    inside[:, 1:-1] &= ~(white[:, 1:-1] & white[:, :-2] & white[:, 2:])

    smoothed = gray.copy()
    for top in range(0, height, _TILE_SIZE):
        for left in range(0, width, _TILE_SIZE):
            rows = slice(top, min(top + _TILE_SIZE, height))
            columns = slice(left, min(left + _TILE_SIZE, width))
            keep = inside[rows, columns]
            if not keep.any():
                continue
            block = _filter_block(padded, gray[rows, columns], top, left, spatial)
            np.copyto(smoothed[rows, columns], block, where=keep)
    return Image.fromarray(smoothed, mode="L")


def _stroke_black(gray: np.ndarray) -> int:
    """A black point deep enough that text comes out ink-black.

    The peak search can collapse onto a spike of already-black pixels and
    hand back zero, and even the true core peak holds little of a stroke's
    mass — so the black point is read off the dark mass itself instead.
    """
    dark = gray[gray < _INK_CEILING]
    if not dark.size:
        return 0
    return int(np.percentile(dark, _FLAT_BLACK_PERCENTILE))


def _whitened(image: Image.Image, flatten: bool) -> Image.Image:
    """The levels-corrected page, flattened first unless told otherwise."""
    if not flatten:
        return whiten_scan(image)
    flat = flatten_scan(image)
    levels = scan_levels(flat, _FLAT_WHITE_PERCENTILE)
    if levels is None:
        return flat
    # The deeper claim wins: the peak shoulder when the histogram offers a
    # real ink peak, the stroke mass when the search collapses to a spike.
    black = max(levels[0], _stroke_black(np.array(flat)))
    return whiten_scan(flat, (black, levels[1]))


def correct_document(
    image: Image.Image, crop_margin: bool = True, flatten: bool = True
) -> Image.Image:
    """Clean a scanned page: flatten its shadows, fix its levels, denoise.

    NAPS2's document correction end to end, led by the illumination flatten
    NAPS2 has no answer to, plus the border removal it leaves as a TODO. The
    margin is measured on the way in and cut off on the way out, because
    correcting the page turns its paper the same pure white as the canvas
    around it — after the fact there is no edge left to find.

    Args:
        image: Input page, color or grayscale.
        crop_margin: Whether to cut the scanner's white canvas off the
            result. Pass False to keep the scan's own dimensions.
        flatten: Whether to divide the illumination out before the levels
            run. Pass False for a document whose light print should come
            through as it is — an answer sheet's shaded table rows would
            bleach to white wherever a shadow crossed them.

    Returns:
        Grayscale ("L") page — the size of the page within the scan, or of
        the whole scan when *crop_margin* is off.
    """
    if not crop_margin:
        return denoise_scan(_whitened(image, flatten))
    rows, columns = _content_box(np.array(image.convert("L")))
    corrected = denoise_scan(_whitened(image, flatten))
    return corrected.crop((columns.start, rows.start, columns.stop, rows.stop))


# --- image cropping helpers ---


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """Rotate by *angle* degrees counterclockwise, growing the canvas to fit."""
    return Image.fromarray(_rotate(np.array(image), angle))


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    rad = math.radians(angle)
    sin_a, cos_a = math.sin(rad), math.cos(rad)
    new_w = round(abs(sin_a) * h + abs(cos_a) * w)
    new_h = round(abs(sin_a) * w + abs(cos_a) * h)

    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    mat[0, 2] += (new_w - w) / 2
    mat[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(
        img,
        mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.empty((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _polygon_to_quad(polygon: PixelPolygon, max_angle: float = 4.0) -> np.ndarray:
    pts = np.array(polygon, dtype=np.int32)
    rect = cv2.minAreaRect(pts)

    if abs(min(rect[2], 90 - rect[2])) > max_angle:
        x, y, w, h = cv2.boundingRect(pts)
        bbox = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
        )
    else:
        bbox = cv2.boxPoints(rect)

    return _order_quad_points(bbox)


def _perspective_transform(pts: np.ndarray) -> tuple[int, int, np.ndarray]:
    w = max(
        int(np.linalg.norm(pts[0] - pts[1])),
        int(np.linalg.norm(pts[2] - pts[3])),
    )
    h = max(
        int(np.linalg.norm(pts[1] - pts[2])),
        int(np.linalg.norm(pts[3] - pts[0])),
    )
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return w, h, cv2.getPerspectiveTransform(pts, dst)


def cut_out_image_by_polygon(image: Image.Image, polygon: PixelPolygon) -> Image.Image:
    """Cut *polygon* out of *image*, deskewed by a perspective transform.

    Everything outside the polygon is left transparent in the returned RGBA
    crop — flatten it (:func:`add_white_background`) before saving to a
    format with no alpha channel.
    """
    if len(polygon) < 4:
        raise ValueError("Polygon must have 4 or more points")

    img = np.array(image.convert("RGBA"))
    pts = _polygon_to_quad(polygon)
    w, h, M = _perspective_transform(pts)

    warped = cv2.warpPerspective(img, M, (w, h))

    poly_np = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    tr_pts = cv2.perspectiveTransform(poly_np, M).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [tr_pts], 255)
    warped[:, :, 3] = cv2.bitwise_and(warped[:, :, 3], mask)

    return Image.fromarray(warped, mode="RGBA")
