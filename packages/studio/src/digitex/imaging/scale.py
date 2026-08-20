"""The two operations that read an image's shape rather than its content.

Fitting a picture inside a box, and flattening what a polygon crop left
transparent onto white. Neither looks at a pixel to decide what to do —
anything that does belongs in one of the neighbouring modules.
"""

import numpy as np
from PIL import Image, ImageOps


def resize_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """Scale *image* to the largest size that fits inside the given bounds.

    The bounds are a target rather than a ceiling: a picture smaller than the
    box is scaled up to meet it, not padded out to it. Aspect ratio survives
    either way, so the result fills one of the two dimensions exactly.
    """
    return ImageOps.contain(
        image, (max_width, max_height), method=Image.Resampling.BILINEAR
    )


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
    rgba = np.array(image.convert("RGBA"))
    # Sliced as 3:4 rather than 3, keeping a trailing axis of length one, so
    # one coverage value broadcasts across all three colour channels.
    coverage = rgba[:, :, 3:4] / 255.0
    blended = rgba[:, :, :3] * coverage + 255 * (1 - coverage)
    return Image.fromarray(blended.astype(np.uint8), mode="RGB")
