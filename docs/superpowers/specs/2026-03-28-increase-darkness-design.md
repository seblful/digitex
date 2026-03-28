# Increase Darkness Feature Design

## Summary

Add a gamma correction method to `SegmentProcessor` that increases contrast of dark content by darkening mid-tones while preserving pure whites and blacks.

## Requirements

- Increase contrast of dark text/lines in processed image segments
- Preserve white background and black content
- Apply at the end of the processing pipeline
- Configurable intensity via gamma parameter

## Design

### New Method: `increase_darkness`

```python
@staticmethod
def increase_darkness(
    image: Image.Image,
    gamma: float = DEFAULT_GAMMA,
) -> Image.Image:
    """Apply gamma correction to darken mid-tones and increase contrast.

    Args:
        image: Input PIL Image (RGBA recommended).
        gamma: Gamma value. Values < 1.0 darken the image, > 1.0 lighten.
            Default 0.8 provides subtle darkening.

    Returns:
        Image with gamma correction applied.
    """
```

**Implementation:**
- Convert to numpy array
- Apply formula: `output = ((input / 255) ** (1 / gamma)) * 255`
- Preserve alpha channel unchanged
- Return PIL Image

### Modified Pipeline

`process()` method updated to:

```
remove_color → remove_bg_threshold → increase_darkness
```

### New Constant

```python
DEFAULT_GAMMA = 0.8
```

## Files Changed

- `src/digitex/core/processors/image.py` — Add `DEFAULT_GAMMA` constant and `increase_darkness()` method, update `process()`
- `tests/test_processors.py` — Add tests for `increase_darkness()`

## Testing

- Test gamma = 0.8 darkens mid-tones
- Test gamma = 1.0 returns unchanged image
- Test alpha channel is preserved
- Test edge cases (gamma < 0, gamma = 0)
