# Background Removal for SegmentHandler

## Summary

Add background removal to `SegmentHandler` as two new preprocess modes (`grabcut` and `threshold`), selectable from the extraction CLI. Both produce transparent RGBA output and force PNG format on save.

## Motivation

The extraction pipeline currently supports `enhance` and `binarize` preprocess modes for cropped question segments. Background removal (making the paper background transparent, keeping only ink/strokes) is needed for use cases where transparent question images are required.

## Design

### ImageProcessor — Low-level helpers

Two new methods on `ImageProcessor` (`src/digitex/core/processors/image.py`):

- **`remove_bg_grabcut(image_bgr: np.ndarray, iter_count: int = 5) -> np.ndarray`** — Uses OpenCV `grabCut` with a rectangular ROI covering the full image. Returns a 4-channel BGRA image. Pixels classified as "probably foreground" or "definite foreground" by GrabCut become opaque; everything else becomes transparent.

- **`remove_bg_threshold(image_bgr: np.ndarray, threshold: int = 240) -> np.ndarray`** — Converts to grayscale, applies a binary inverse threshold. Pixels with value above `threshold` become transparent; pixels at or below `threshold` become opaque. Returns a 4-channel BGRA image.

Both are pure OpenCV — no new dependencies.

### SegmentHandler — Two new methods

Two new methods on `SegmentHandler` (`src/digitex/core/processors/image.py`), following the same thin-wrapper pattern as `enhance()` and `binarize()`:

- **`remove_bg_grabcut(segment_bgr: np.ndarray) -> np.ndarray`** — Delegates to `self._processor.remove_bg_grabcut(segment_bgr)`.
- **`remove_bg_threshold(segment_bgr: np.ndarray) -> np.ndarray`** — Delegates to `self._processor.remove_bg_threshold(segment_bgr)`.

Neither method runs the `preprocess()` pipeline (color removal, denoising, CLAHE). Background removal is a standalone operation — the document cleanup pipeline is irrelevant when the goal is to separate foreground from background.

### PageExtractor — Handle new modes

In `PageExtractor.extract()` (`src/digitex/extractors/page_extractor.py`), expand the preprocess branching:

```
"enhance"    → SegmentHandler.enhance()        → grayscale → RGB PIL Image
"binarize"   → SegmentHandler.binarize()       → grayscale → RGB PIL Image
"grabcut"    → SegmentHandler.remove_bg_grabcut()  → BGRA → RGBA PIL Image
"threshold"  → SegmentHandler.remove_bg_threshold() → BGRA → RGBA PIL Image
None         → no processing                    → original RGB PIL Image
```

When `preprocess` is `"grabcut"` or `"threshold"`, the `image_format` is overridden to `"png"` for that save call. Other modes continue to use the configured `image_format`.

### CLI

The `preprocess` parameter in `extraction/run.py` accepts `str | None`. Valid values expand from `enhance | binarize` to `enhance | binarize | grabcut | threshold`. The help text is updated accordingly. No type changes needed.

### Tests

- **`test_remove_bg_grabcut_returns_bgra`** — Verify 4-channel output with non-trivial alpha mask (not all opaque or all transparent).
- **`test_remove_bg_threshold_returns_bgra`** — Verify 4-channel output where bright pixels are transparent and dark pixels are opaque.
- **`test_segment_handler_remove_bg_grabcut_delegates`** — Verify delegation to ImageProcessor.
- **`test_segment_handler_remove_bg_threshold_delegates`** — Verify delegation to ImageProcessor.
- **`test_page_extractor_grabcut_forces_png`** — Integration test: preprocess="grabcut" saves as PNG regardless of image_format setting.
- **`test_page_extractor_threshold_forces_png`** — Integration test: preprocess="threshold" saves as PNG regardless of image_format setting.

## Files changed

| File | Change |
|---|---|
| `src/digitex/core/processors/image.py` | Add `remove_bg_grabcut` and `remove_bg_threshold` to `ImageProcessor` and `SegmentHandler` |
| `src/digitex/extractors/page_extractor.py` | Add `grabcut`/`threshold` branches, PNG format override |
| `extraction/run.py` | Update help text for `preprocess` parameter |
| `tests/test_image_processor.py` | Add unit tests for new ImageProcessor methods |
| `tests/test_page_extractor.py` | Add integration tests for new preprocess modes |
