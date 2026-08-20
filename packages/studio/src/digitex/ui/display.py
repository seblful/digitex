"""Making the window draw at the display's real resolution.

Windows hands a process that has not claimed to understand high-DPI displays a
96-DPI canvas and lets the compositor stretch the result. Text drawn that way
comes out soft and washed out — the whole window looks muted beside every other
application on the machine.

Saying so once, before the first Tk window exists, is the entire fix. The factor
that call reports back is then applied to the sizes the windows are written in,
which are all written for a 100% display, so the layout keeps its proportions at
any scaling.

Nothing here builds a widget: this has to run before there is one.
"""

from __future__ import annotations

import os

import structlog

logger = structlog.get_logger()

# What Windows calls 100%: the DPI every hardcoded pixel size in this package
# assumes it is being drawn at.
BASE_DPI = 96

# Under this a display is not really scaled, and rounding every width up would
# buy a bigger window and nothing else.
MIN_MEANINGFUL_SCALE = 1.05


def scale_from_dpi(dpi: float) -> float:
    """The factor a 96-DPI layout must grow by to fill *dpi* the same way.

    A display that reports nothing — or a scaling too slight to be worth
    resizing for — comes back as 1.0, meaning "leave the sizes alone".
    """
    if dpi <= 0:
        return 1.0
    scale = dpi / BASE_DPI
    return 1.0 if scale < MIN_MEANINGFUL_SCALE else scale


def scaled(value: int, scale: float) -> int:
    """A size written in 96-DPI pixels, at *scale*."""
    return round(value * scale)


def enable_dpi_awareness() -> float:
    """Declare this process DPI-aware and report the display's scale factor.

    Must run before the first Tk window is created: Tk reads the screen's DPI as
    its interpreter starts, and a process that turns aware after that has
    already been measured at 96.

    Returns 1.0 on anything that is not Windows, and on a Windows too old to be
    asked.
    """
    if os.name != "nt":
        return 1.0

    import ctypes

    try:
        # 1 = system-DPI aware. Per-monitor awareness would oblige the window to
        # handle a DPI change as it is dragged between screens, which Tk cannot.
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        # Either the host process already declared it, or this Windows has no
        # shcore. The older call is the fallback, and losing that one too is not
        # fatal — the window merely looks the way it did before.
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            logger.debug("Could not declare DPI awareness")

    try:
        dpi = float(ctypes.windll.user32.GetDpiForSystem())
    except (AttributeError, OSError):
        return 1.0

    return scale_from_dpi(dpi)
