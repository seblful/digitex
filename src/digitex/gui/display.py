"""Making the window draw at the display's real resolution.

Windows scales a process that has not said it understands high-DPI displays:
it is handed a 96-DPI canvas and the result is stretched by the compositor.
Text drawn that way is blurred and washed out — the whole window looks muted
next to every other application on the machine.

Saying so once, before the first Tk window exists, is the whole fix. The
factor it reports back is then applied to the sizes that were written in
96-DPI pixels, so the window keeps its proportions at any scaling.
"""

from __future__ import annotations

import os

import structlog

logger = structlog.get_logger()

# What Windows calls 100%: the DPI every hardcoded pixel size here assumes.
BASE_DPI = 96

# Below this a display is not really scaled, and rounding the widths up would
# only make the window bigger for nothing.
MIN_MEANINGFUL_SCALE = 1.05


def scale_from_dpi(dpi: float) -> float:
    """The factor a 96-DPI layout must grow by to fill *dpi* the same way."""
    if dpi <= 0:
        return 1.0
    scale = dpi / BASE_DPI
    return 1.0 if scale < MIN_MEANINGFUL_SCALE else scale


def scaled(value: int, scale: float) -> int:
    """A size written in 96-DPI pixels, at *scale*."""
    return round(value * scale)


def enable_dpi_awareness() -> float:
    """Declare this process DPI-aware and report the display's scale factor.

    Must run before the first Tk window is created: Tk reads the screen's DPI
    when its interpreter starts, and a process that becomes aware afterwards
    has already been measured at 96.

    Returns 1.0 on anything but Windows, and on a Windows too old to ask.
    """
    if os.name != "nt":
        return 1.0

    import ctypes

    try:
        # 1 = system-DPI aware. Per-monitor would need the window to handle a
        # DPI change as it moves between screens, which Tk does not do.
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        # Already set by the host process, or a Windows without shcore. The
        # older call is the fallback, and failing it is not fatal — the window
        # just looks the way it did before.
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            logger.debug("Could not declare DPI awareness")

    try:
        dpi = float(ctypes.windll.user32.GetDpiForSystem())
    except (AttributeError, OSError):
        return 1.0

    return scale_from_dpi(dpi)
