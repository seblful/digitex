"""Tests for the image processors: cropping, rotation, flattening, resizing."""

import numpy as np
import pytest
from PIL import Image

from digitex.domain.entities import PixelPolygon
from digitex.imaging import (
    add_white_background,
    correct_document,
    denoise_scan,
    flatten_scan,
    resize_image,
    rotate_image,
    scan_levels,
    stack_vertically,
    stacked_layout,
    whiten_scan,
)
from digitex.imaging.crop import (
    _order_quad_points,
    _perspective_transform,
    _polygon_to_quad,
    cut_out_image_by_polygon,
)
from digitex.imaging.levels import _levels_from, content_box

_INK_ROWS = (slice(10, 14), slice(20, 100))
_PAPER_PATCH = (slice(80, 110), slice(40, 100))


def _scanned_page(paper: int = 210, ink: int = 30, margin: int = 12) -> Image.Image:
    """Grainy paper carrying grainy ink, beside a saturated scan margin.

    The grain matters: the correction reads peak *shapes* out of the
    histogram, so a page painted in three flat tones tells it nothing that a
    real scan would. Widen *margin* to let the scanner's canvas outnumber the
    paper, which is what trips the plain peak search.
    """
    rng = np.random.default_rng(0)
    pixels = rng.normal(paper, 3.5, (120, 120))
    for top in (10, 40, 70):
        pixels[top : top + 4, 20:100] = rng.normal(ink, 6, (4, 80))
    pixels[:, :margin] = 255
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="L")


_SHADOW_HALF = (slice(320, 576), slice(16, 150))
_LIT_HALF = (slice(320, 576), slice(490, 624))
_SHADOWED_INK = (slice(160, 164), slice(16, 150))
_LIT_INK = (slice(160, 164), slice(490, 624))


def _shadowed_page(
    paper: int = 220, ink: int = 30, dimming: float = 0.6
) -> Image.Image:
    """A page whose left side sits under a gutter's shadow.

    Five background tiles across, with the shadow shaped like the archive's
    gutters: a flat floor two tiles wide, ramping back to full light over
    the next — wide enough that the background grid can follow it, which is
    the scale the flatten was calibrated for. The illumination multiplies
    the whole scene, ink included, which is what makes division the right
    correction to test for.
    """
    rng = np.random.default_rng(1)
    pixels = rng.normal(paper, 3.5, (640, 640))
    for top in (48, 160, 272):
        pixels[top : top + 4, 16:624] = rng.normal(ink, 6, (4, 608))
    light = np.interp(np.arange(640), [256, 384], [dimming, 1.0])
    return Image.fromarray(
        np.clip(pixels * light[np.newaxis, :], 0, 255).astype(np.uint8), mode="L"
    )


class TestRotateImage:
    def test_ninety_degrees_swaps_the_dimensions(self) -> None:
        img = Image.new("RGB", (100, 50), color="white")

        result = rotate_image(img, 90.0)

        assert result.size == (50, 100)

    def test_the_canvas_grows_to_hold_the_rotated_image(self) -> None:
        """Nothing may be cut off — a tilted crop's corners stay inside."""
        img = Image.new("RGB", (100, 100), color="white")

        result = rotate_image(img, 45.0)

        assert result.size == (141, 141)


class TestAddWhiteBackground:
    def test_transparent_becomes_white(self) -> None:
        img = Image.new("RGBA", (10, 10), color=(100, 100, 100, 0))

        result = add_white_background(img)

        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (255, 255, 255)

    def test_opaque_unchanged(self) -> None:
        img = Image.new("RGBA", (10, 10), color=(50, 100, 150, 255))

        result = add_white_background(img)

        assert result.getpixel((0, 0)) == (50, 100, 150)

    def test_fully_transparent_image_is_all_white(self) -> None:
        img = Image.new("RGBA", (10, 10), color=(0, 0, 0, 0))
        result = add_white_background(img)
        assert result.mode == "RGB"
        assert result.size == (10, 10)
        assert result.getpixel((5, 5)) == (255, 255, 255)

    def test_partial_transparency_blends_toward_white(self) -> None:
        img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        result = add_white_background(img)
        pixel = result.getpixel((0, 0))
        # RGB, so a 3-tuple — narrowed because getpixel also describes the
        # single-band and out-of-bounds cases this image cannot produce.
        assert isinstance(pixel, tuple)
        red, green, blue = pixel
        assert red > 127
        assert green < 128
        assert blue < 128


class TestResizeImage:
    def test_pads_smaller_image_to_max_dimensions(self) -> None:
        img = Image.new("RGB", (100, 100), color="red")
        result = resize_image(img, 200, 200)
        assert result.size == (200, 200)

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ((400, 200), (200, 100)),
            ((200, 400), (100, 200)),
        ],
        ids=["width-limited", "height-limited"],
    )
    def test_preserves_aspect_ratio(
        self, source: tuple[int, int], expected: tuple[int, int]
    ) -> None:
        img = Image.new("RGB", source, color="red")
        result = resize_image(img, 200, 200)
        assert result.size == expected

    def test_landscape_shrinks_to_fit(self) -> None:
        img = Image.new("RGB", (200, 100), color="red")
        result = resize_image(img, 100, 100)
        assert result.size == (100, 50)

    def test_portrait_shrinks_to_fit(self) -> None:
        img = Image.new("RGB", (100, 200), color="red")
        result = resize_image(img, 100, 100)
        assert result.size == (50, 100)


class TestStackedLayout:
    """Where the pieces of a question printed across a page break end up."""

    def test_pieces_follow_one_another_down_the_stack(self) -> None:
        size, positions = stacked_layout([(100, 40), (100, 60)], gap=10)

        assert positions == [(0, 0), (0, 50)]
        assert size == (100, 110)

    def test_the_canvas_takes_the_widest_piece(self) -> None:
        size, _ = stacked_layout([(60, 10), (100, 10)])

        assert size == (100, 20)

    def test_an_offset_moves_a_piece_against_the_one_above_it(self) -> None:
        size, positions = stacked_layout(
            [(100, 40), (100, 40)], offsets=[(0, 0), (8, -10)]
        )

        assert positions == [(0, 0), (8, 30)]
        # The nudged piece hangs 8px off the right, and 10px of the seam closed.
        assert size == (108, 70)

    def test_a_piece_nudged_left_of_the_first_still_lands_on_the_canvas(self) -> None:
        _, positions = stacked_layout(
            [(100, 40), (100, 40)], offsets=[(0, 0), (-20, 0)]
        )

        assert positions == [(20, 0), (0, 40)]

    def test_a_nudge_carries_the_pieces_below_it_along(self) -> None:
        """Lining up one seam must not move the seams above it."""
        _, positions = stacked_layout(
            [(50, 10), (50, 10), (50, 10)], offsets=[(0, 0), (5, 0), (0, 0)]
        )

        assert positions == [(0, 0), (5, 10), (5, 20)]

    def test_the_first_piece_has_nothing_to_sit_against(self) -> None:
        _, positions = stacked_layout([(50, 10), (50, 10)], offsets=[(90, 90), (0, 0)])

        assert positions == [(0, 0), (0, 10)]


class TestStackVertically:
    def test_the_pieces_keep_their_own_size(self) -> None:
        stacked = stack_vertically(
            [Image.new("RGB", (80, 30), "red"), Image.new("RGB", (60, 40), "blue")],
            gap=6,
        )

        assert stacked.size == (80, 76)
        assert stacked.getpixel((10, 10)) == (255, 0, 0)
        assert stacked.getpixel((10, 50)) == (0, 0, 255)
        # The gap, and the padding beside the narrower piece, are white.
        assert stacked.getpixel((10, 32)) == (255, 255, 255)
        assert stacked.getpixel((70, 50)) == (255, 255, 255)

    def test_one_piece_is_handed_back_untouched(self) -> None:
        """A whole question is not a stack, and must not be padded like one."""
        only = Image.new("RGB", (40, 20), "red")

        assert stack_vertically([only], gap=6) is only

    def test_an_offset_places_the_piece_where_it_was_lined_up(self) -> None:
        stacked = stack_vertically(
            [Image.new("RGB", (40, 20), "red"), Image.new("RGB", (40, 20), "blue")],
            gap=0,
            offsets=[(0, 0), (10, 0)],
        )

        assert stacked.size == (50, 40)
        assert stacked.getpixel((15, 30)) == (0, 0, 255)
        assert stacked.getpixel((5, 30)) == (255, 255, 255)

    def test_nothing_to_stack_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Nothing to stack"):
            stack_vertically([])


class TestScanLevels:
    def test_the_white_point_lands_inside_the_paper_peak(self) -> None:
        """A shoulder, not an edge — the paper's darker half clips too."""
        black, white = scan_levels(_scanned_page(paper=210)) or (0, 0)

        assert 200 <= white < 210
        assert black < white

    def test_the_black_point_lands_inside_the_ink_peak(self) -> None:
        levels = scan_levels(_scanned_page(paper=210, ink=30))

        assert levels is not None
        assert 30 <= levels[0] <= 50

    def test_both_points_follow_the_page(self) -> None:
        pale = scan_levels(_scanned_page(paper=240, ink=20))
        dim = scan_levels(_scanned_page(paper=210, ink=30))

        assert pale is not None
        assert dim is not None
        assert pale[1] > dim[1]
        assert pale[0] < dim[0]

    def test_a_featureless_page_offers_no_pair(self) -> None:
        """One flat tone gives no ink peak to stretch away from."""
        assert scan_levels(Image.new("L", (60, 60), color=128)) is None

    def test_a_wide_scan_margin_does_not_pass_for_paper(self) -> None:
        """Left to itself the search picks the canvas and corrects nothing."""
        page = _scanned_page(paper=210, margin=60)

        assert _levels_from(page.histogram())[1] == 255
        assert scan_levels(page) == (34, 207)

    def test_a_narrow_margin_is_left_to_the_plain_search(self) -> None:
        """The second look is for pages that need it, not for every page."""
        page = _scanned_page(paper=210, margin=12)

        assert scan_levels(page) == _levels_from(page.histogram())[0]


class TestContentBox:
    def test_the_scan_margin_is_cut_off(self) -> None:
        pixels = np.full((60, 60), 255, dtype=np.uint8)
        pixels[10:50, 20:55] = 200

        rows, columns = content_box(pixels)

        assert (rows.start, rows.stop) == (10, 50)
        assert (columns.start, columns.stop) == (20, 55)

    def test_a_page_with_no_margin_is_kept_whole(self) -> None:
        pixels = np.full((60, 60), 200, dtype=np.uint8)

        rows, columns = content_box(pixels)

        assert pixels[rows, columns].shape == (60, 60)

    def test_an_all_white_scan_is_kept_whole(self) -> None:
        """Nothing to find, so nothing is cut — the caller still gets a page."""
        pixels = np.full((60, 60), 255, dtype=np.uint8)

        rows, columns = content_box(pixels)

        assert pixels[rows, columns].shape == (60, 60)


class TestWhitenScan:
    def test_paper_and_margin_come_out_pure_white(self) -> None:
        result = np.array(whiten_scan(_scanned_page(paper=210)))

        assert result[0, 0] == 255
        assert int(np.median(result[_PAPER_PATCH])) == 255

    def test_ink_is_driven_to_black(self) -> None:
        result = np.array(whiten_scan(_scanned_page(paper=210, ink=30)))

        assert int(np.median(result[_INK_ROWS])) == 0

    def test_a_washed_out_page_gets_its_ink_deepened(self) -> None:
        """The black point earns its keep here: ink at 90 still reaches 0."""
        page = _scanned_page(paper=240, ink=90)

        result = np.array(whiten_scan(page))

        assert int(np.median(np.array(page)[_INK_ROWS])) > 80
        assert int(np.median(result[_INK_ROWS])) == 0

    def test_explicit_levels_override_the_page(self) -> None:
        page = _scanned_page(paper=210)

        result = np.array(whiten_scan(page, levels=(0, 255)))

        assert np.array_equal(result, np.array(page))

    def test_a_page_it_declines_to_correct_is_returned_as_is(self) -> None:
        flat = Image.new("L", (60, 60), color=128)

        assert np.array(whiten_scan(flat)).tolist() == np.array(flat).tolist()

    def test_color_input_comes_back_grayscale(self) -> None:
        result = whiten_scan(Image.new("RGB", (10, 10), color=(200, 200, 200)))

        assert result.mode == "L"
        assert result.size == (10, 10)


class TestFlattenScan:
    def test_shadowed_paper_lifts_to_the_lit_papers_level(self) -> None:
        page = np.array(_shadowed_page(paper=220, dimming=0.6))
        assert np.median(page[_LIT_HALF]) - np.median(page[_SHADOW_HALF]) > 60

        result = np.array(flatten_scan(Image.fromarray(page, mode="L")))

        lit, shadowed = np.median(result[_LIT_HALF]), np.median(result[_SHADOW_HALF])
        assert abs(lit - shadowed) < 8

    def test_ink_in_the_shadow_recovers_its_tone(self) -> None:
        result = np.array(flatten_scan(_shadowed_page(ink=30, dimming=0.6)))

        lit, shadowed = np.median(result[_LIT_INK]), np.median(result[_SHADOWED_INK])
        assert abs(lit - shadowed) < 15

    def test_a_uniform_page_is_left_alone(self) -> None:
        page = Image.new("L", (384, 384), color=200)

        assert np.array_equal(np.array(flatten_scan(page)), np.array(page))

    def test_a_dark_figure_is_not_taken_for_a_shadow(self) -> None:
        """A figure spanning whole tiles borrows its neighbours' paper."""
        rng = np.random.default_rng(2)
        pixels = np.clip(rng.normal(220, 3.5, (640, 640)), 0, 255)
        pixels[192:448, 192:448] = 60

        result = np.array(flatten_scan(Image.fromarray(pixels.astype(np.uint8))))

        assert np.median(result[256:384, 256:384]) < 90

    def test_color_input_comes_back_grayscale(self) -> None:
        result = flatten_scan(Image.new("RGB", (60, 60), color=(200, 200, 200)))

        assert result.mode == "L"
        assert result.size == (60, 60)


class TestDenoiseScan:
    def test_grain_in_the_paper_averages_away(self) -> None:
        rng = np.random.default_rng(0)
        grain = rng.integers(236, 245, size=(60, 60), dtype=np.uint8)

        result = np.array(denoise_scan(Image.fromarray(grain, mode="L")))

        assert result[10:50, 10:50].std() < grain[10:50, 10:50].std() / 2

    def test_the_edge_between_ink_and_paper_survives(self) -> None:
        """A blur would smear this step; weighting by tone must not."""
        pixels = np.full((60, 60), 255, dtype=np.uint8)
        pixels[:, 30:] = 0

        result = np.array(denoise_scan(Image.fromarray(pixels, mode="L")))

        assert result[30, 29] > 250
        assert result[30, 30] < 5

    def test_the_filters_own_reach_is_left_alone(self) -> None:
        """Half a window from any edge there are no neighbours to average."""
        rng = np.random.default_rng(0)
        grain = rng.integers(236, 245, size=(60, 60), dtype=np.uint8)

        result = np.array(denoise_scan(Image.fromarray(grain, mode="L")))

        assert np.array_equal(result[:8], grain[:8])
        assert np.array_equal(result[:, -7:], grain[:, -7:])

    def test_a_run_of_white_is_kept_pure(self) -> None:
        """White between white is paper, and paper is left at 255."""
        pixels = np.full((60, 60), 250, dtype=np.uint8)
        pixels[20, 20:40] = 255
        pixels[40, 30] = 255

        result = np.array(denoise_scan(Image.fromarray(pixels, mode="L")))

        assert result[20, 30] == 255
        assert result[40, 30] < 255

    def test_color_input_comes_back_grayscale(self) -> None:
        result = denoise_scan(Image.new("RGB", (60, 60), color=(200, 200, 200)))

        assert result.mode == "L"
        assert result.size == (60, 60)


class TestCorrectDocument:
    def test_the_page_comes_out_clean(self) -> None:
        page = _scanned_page(paper=210, ink=30)

        result = np.array(correct_document(page, crop_margin=False))

        # A level short of pure: averaging a white pixel against the grain
        # still under the white point costs it that much.
        assert int(np.median(result[_PAPER_PATCH])) >= 254
        assert int(np.median(result[_INK_ROWS])) == 0

    def test_a_gutter_shadow_reads_as_the_same_paper(self) -> None:
        """The whole point of flattening first.

        The shadow half of the page comes out as bright as the lit half,
        not stretched toward black.
        """
        page = _shadowed_page(paper=220, dimming=0.6)

        result = np.array(correct_document(page, crop_margin=False))

        shadowed = np.median(result[_SHADOW_HALF])
        lit = np.median(result[_LIT_HALF])
        assert shadowed >= 240
        assert abs(lit - shadowed) <= 8

    def test_ink_comes_out_ink_black(self) -> None:
        """Solid strokes on both halves.

        The flatten restores a shadowed stroke's tone and the black point
        then reaches it.
        """
        page = _shadowed_page(paper=220, ink=30, dimming=0.6)

        result = np.array(correct_document(page, crop_margin=False))

        assert np.median(result[_LIT_INK]) == 0
        assert np.median(result[_SHADOWED_INK]) == 0

    def test_flatten_off_keeps_the_plain_correction(self) -> None:
        """The opt-out for answer sheets: no flatten, NAPS2's levels alone."""
        page = _shadowed_page(paper=220, dimming=0.6)

        result = np.array(correct_document(page, crop_margin=False, flatten=False))

        shadowed = np.median(result[_SHADOW_HALF])
        lit = np.median(result[_LIT_HALF])
        assert lit - shadowed > 40

    def test_the_scan_margin_is_gone(self) -> None:
        page = _scanned_page(paper=210, margin=12)

        result = correct_document(page)

        assert result.mode == "L"
        assert result.size == (108, 120)

    def test_keeping_the_margin_keeps_the_scans_dimensions(self) -> None:
        page = _scanned_page(paper=210, margin=12)

        assert correct_document(page, crop_margin=False).size == page.size

    def test_cropping_does_not_disturb_the_pixels_it_keeps(self) -> None:
        """The margin comes off after correction, so the page is untouched."""
        page = _scanned_page(paper=210, margin=12)

        whole = np.array(correct_document(page, crop_margin=False))
        cropped = np.array(correct_document(page))

        assert np.array_equal(cropped, whole[:, 12:])


class TestCutOutImageByPolygon:
    def test_order_quad_points(self) -> None:
        pts = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)
        ordered = _order_quad_points(pts)
        assert ordered.shape == (4, 2)
        assert ordered[0, 0] < ordered[2, 0]

    def test_order_quad_points_degenerate_all_same(self) -> None:
        pts = np.array([[50, 50], [50, 50], [50, 50], [50, 50]], dtype=np.float32)
        ordered = _order_quad_points(pts)
        assert ordered.shape == (4, 2)

    def test_order_quad_points_collinear(self) -> None:
        pts = np.array([[0, 0], [50, 0], [100, 0], [150, 0]], dtype=np.float32)
        ordered = _order_quad_points(pts)
        assert ordered.shape == (4, 2)

    @pytest.mark.parametrize(
        "polygon",
        [
            [(10, 10), (50, 10), (50, 50), (10, 50)],
            [(10, 10), (50, 15), (48, 50), (12, 48)],
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            [(10, 5), (60, 15), (50, 70), (0, 60)],
        ],
        ids=["rectangle", "skewed", "square", "rotated"],
    )
    def test_polygon_to_quad_returns_four_points(
        self, polygon: list[tuple[int, int]]
    ) -> None:
        quad = _polygon_to_quad(PixelPolygon(polygon))
        assert quad.shape == (4, 2)

    def test_perspective_transform_dimensions(self) -> None:
        pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        w, h, _ = _perspective_transform(pts)
        assert w == 100
        assert h == 50

    def test_perspective_transform_trapezoid(self) -> None:
        pts = np.array([[10, 0], [90, 0], [100, 50], [0, 50]], dtype=np.float32)
        w, h, _ = _perspective_transform(pts)
        assert w >= 90
        assert h == 50

    def test_cut_out_requires_four_or_more_points(self) -> None:
        img = Image.new("RGB", (100, 100), color="white")
        with pytest.raises(ValueError, match="Polygon must have 4 or more points"):
            cut_out_image_by_polygon(img, PixelPolygon([(10, 10), (20, 20)]))

    @pytest.mark.parametrize(
        "polygon",
        [
            [(10, 10), (190, 10), (190, 190), (10, 190)],
            [
                (20, 20),
                (50, 20),
                (80, 20),
                (80, 50),
                (80, 80),
                (50, 80),
                (20, 80),
                (20, 50),
            ],
            [(25, 25), (75, 20), (80, 50), (70, 80), (30, 75), (20, 50)],
        ],
        ids=["rectangle", "many-points", "irregular-hexagon"],
    )
    def test_cut_out_returns_rgba_crop(self, polygon: list[tuple[int, int]]) -> None:
        img = Image.new("RGB", (300, 300), color="white")
        result = cut_out_image_by_polygon(img, PixelPolygon(polygon))
        assert result.mode == "RGBA"
        assert result.size[0] > 0
        assert result.size[1] > 0
