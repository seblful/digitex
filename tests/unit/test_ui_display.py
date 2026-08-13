"""Tests for the high-DPI sizing rule.

Declaring DPI awareness is a Windows call that cannot be made twice in a
process, so what is checked here is the arithmetic it feeds: how a layout
written in 96-DPI pixels is resized for the display it actually lands on.
"""

from digitex.ui.display import scale_from_dpi, scaled


class TestScaleFromDpi:
    def test_an_unscaled_display_changes_nothing(self) -> None:
        assert scale_from_dpi(96) == 1.0

    def test_a_scaled_display_reports_its_factor(self) -> None:
        assert scale_from_dpi(120) == 1.25  # Windows at 125%
        assert scale_from_dpi(144) == 1.5  # at 150%

    def test_a_hair_over_100_percent_is_not_worth_resizing_for(self) -> None:
        assert scale_from_dpi(100) == 1.0

    def test_a_display_that_reports_nothing_is_treated_as_unscaled(self) -> None:
        assert scale_from_dpi(0) == 1.0


class TestScaled:
    def test_sizes_grow_with_the_display(self) -> None:
        assert scaled(400, 1.25) == 500

    def test_sizes_come_back_as_whole_pixels(self) -> None:
        size = scaled(230, 1.5)

        assert size == 345
        assert isinstance(size, int)
