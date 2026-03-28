# White Background Feature Design

## Summary

Add a method to `SegmentProcessor` that composites the RGBA image onto a white background, returning an RGB image suitable for JPG format.

## Requirements

- Replace transparent background with white
- Return RGB image (no alpha channel)
- Make output suitable for JPG saving

## Design

### New Method: `add_white_background`

```python
@staticmethod
def add_white_background(image: Image.Image) -> Image.Image:
    """Composite RGBA image onto white background.

    Args:
        image: Input PIL Image (RGBA recommended).

    Returns:
        RGB image with white background replacing transparency.
    """
```

**Implementation:**
- Create white RGB background same size as input
- Composite input onto white background using alpha channel
- Return RGB image

### Modified Pipeline

`process()` method updated to:

```
remove_color → remove_bg_threshold → increase_darkness → add_white_background
```

## Files Changed

- `src/digitex/core/processors/image.py` — Add `add_white_background()` method, update `process()`
- `tests/test_processors.py` — Add tests for `add_white_background()`

## Testing

- Test transparent areas become white
- Test returns RGB mode (no alpha)
- Test opaque pixels unchanged
