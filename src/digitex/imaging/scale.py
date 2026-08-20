"""Fitting an image to a box, and giving it something to sit on.

The two operations that care about an image's size rather than its
content.
"""

import numpy as np
from PIL import Image, ImageOps


def resize_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
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
    img = np.array(image.convert("RGBA"))
    alpha = img[:, :, 3:4] / 255.0
    white_bg = np.ones_like(img[:, :, :3]) * 255
    rgb = img[:, :, :3] * alpha + white_bg * (1 - alpha)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")
