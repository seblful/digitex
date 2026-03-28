# Background Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `grabcut` and `threshold` background removal modes to the extraction pipeline, producing transparent PNG images.

**Architecture:** Two new low-level methods on `ImageProcessor` (GrabCut and white-threshold), thin wrappers on `SegmentHandler`, expanded branching in `PageExtractor._crop_and_save()`, and updated CLI help text. No new dependencies.

**Tech Stack:** Python 3.13, OpenCV, numpy, Pillow, pytest

---

### Task 1: Add `remove_bg_grabcut` to ImageProcessor

**Files:**
- Modify: `src/digitex/core/processors/image.py` (after line 249, before `class ImageCropper`)
- Test: `tests/test_processors.py`

- [ ] **Step 1: Write the failing test**

Add to `class TestImageProcessor` in `tests/test_processors.py`:

```python
def test_remove_bg_grabcut_returns_bgra(self) -> None:
    """Test remove_bg_grabcut returns 4-channel BGRA image with non-trivial alpha."""
    processor = ImageProcessor()
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    img[20:80, 20:80] = [50, 50, 50]
    result = processor.remove_bg_grabcut(img)

    assert result.shape == (100, 100, 4)
    assert result.dtype == np.uint8
    alpha = result[:, :, 3]
    assert not np.all(alpha == 0)
    assert not np.all(alpha == 255)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_processors.py::TestImageProcessor::test_remove_bg_grabcut_returns_bgra -v`
Expected: FAIL with `AttributeError: 'ImageProcessor' object has no attribute 'remove_bg_grabcut'`

- [ ] **Step 3: Write minimal implementation**

Add to `ImageProcessor` in `src/digitex/core/processors/image.py`, after `apply_morphology` (after line 249):

```python
    @staticmethod
    def remove_bg_grabcut(
        image_bgr: np.ndarray,
        iter_count: int = 5,
    ) -> np.ndarray:
        """Remove background using GrabCut algorithm.

        Args:
            image_bgr: Input image in BGR format.
            iter_count: Number of GrabCut iterations.

        Returns:
            4-channel BGRA image with transparent background.
        """
        mask = np.zeros(image_bgr.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (1, 1, image_bgr.shape[1] - 2, image_bgr.shape[0] - 2)

        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)

        binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        b, g, r = cv2.split(image_bgr)
        return cv2.merge([b, g, r, binary_mask])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_processors.py::TestImageProcessor::test_remove_bg_grabcut_returns_bgra -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_processors.py src/digitex/core/processors/image.py
git commit -m "feat: add remove_bg_grabcut to ImageProcessor"
```

---

### Task 2: Add `remove_bg_threshold` to ImageProcessor

**Files:**
- Modify: `src/digitex/core/processors/image.py` (after `remove_bg_grabcut`)
- Test: `tests/test_processors.py`

- [ ] **Step 1: Write the failing test**

Add to `class TestImageProcessor` in `tests/test_processors.py`:

```python
def test_remove_bg_threshold_returns_bgra(self) -> None:
    """Test remove_bg_threshold returns 4-channel BGRA with transparent bright pixels."""
    processor = ImageProcessor()
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    img[20:80, 20:80] = [50, 50, 50]
    result = processor.remove_bg_threshold(img)

    assert result.shape == (100, 100, 4)
    assert result.dtype == np.uint8
    alpha = result[:, :, 3]
    assert alpha[0, 0] == 0
    assert alpha[50, 50] == 255
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_processors.py::TestImageProcessor::test_remove_bg_threshold_returns_bgra -v`
Expected: FAIL with `AttributeError: 'ImageProcessor' object has no attribute 'remove_bg_threshold'`

- [ ] **Step 3: Write minimal implementation**

Add to `ImageProcessor` in `src/digitex/core/processors/image.py`, after `remove_bg_grabcut`:

```python
    @staticmethod
    def remove_bg_threshold(
        image_bgr: np.ndarray,
        threshold: int = 240,
    ) -> np.ndarray:
        """Remove background using white-pixel threshold.

        Pixels brighter than the threshold become transparent.
        Pixels at or below the threshold become opaque.

        Args:
            image_bgr: Input image in BGR format.
            threshold: Brightness threshold (0-255).

        Returns:
            4-channel BGRA image with transparent background.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        b, g, r = cv2.split(image_bgr)
        return cv2.merge([b, g, r, binary_mask])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_processors.py::TestImageProcessor::test_remove_bg_threshold_returns_bgra -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_processors.py src/digitex/core/processors/image.py
git commit -m "feat: add remove_bg_threshold to ImageProcessor"
```

---

### Task 3: Add background removal methods to SegmentHandler

**Files:**
- Modify: `src/digitex/core/processors/image.py` (in `SegmentHandler` class, after line 451)
- Test: `tests/test_processors.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_processors.py`:

```python
class TestSegmentHandler:
    """Test suite for SegmentHandler class."""

    def test_remove_bg_grabcut_delegates(self) -> None:
        """Test remove_bg_grabcut delegates to ImageProcessor."""
        from unittest.mock import MagicMock, patch

        handler = SegmentHandler()
        img = np.ones((50, 50, 3), dtype=np.uint8) * 100

        with patch.object(handler._processor, "remove_bg_grabcut", wraps=handler._processor.remove_bg_grabcut) as mock:
            result = handler.remove_bg_grabcut(img)
            mock.assert_called_once_with(img)
            assert result.shape == (50, 50, 4)

    def test_remove_bg_threshold_delegates(self) -> None:
        """Test remove_bg_threshold delegates to ImageProcessor."""
        from unittest.mock import patch

        handler = SegmentHandler()
        img = np.ones((50, 50, 3), dtype=np.uint8) * 100

        with patch.object(handler._processor, "remove_bg_threshold", wraps=handler._processor.remove_bg_threshold) as mock:
            result = handler.remove_bg_threshold(img)
            mock.assert_called_once_with(img)
            assert result.shape == (50, 50, 4)
```

Also add the import at the top of `tests/test_processors.py`:

```python
from digitex.core.processors import FileProcessor, ImageProcessor, SegmentHandler
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_processors.py::TestSegmentHandler -v`
Expected: FAIL with `AttributeError: 'SegmentHandler' object has no attribute 'remove_bg_grabcut'`

- [ ] **Step 3: Write minimal implementation**

Add to `SegmentHandler` in `src/digitex/core/processors/image.py`, after `binarize()`:

```python
    def remove_bg_grabcut(self, segment_bgr: np.ndarray) -> np.ndarray:
        """Remove background using GrabCut algorithm.

        Args:
            segment_bgr: Input segment in BGR format.

        Returns:
            4-channel BGRA image with transparent background.
        """
        return self._processor.remove_bg_grabcut(segment_bgr)

    def remove_bg_threshold(self, segment_bgr: np.ndarray) -> np.ndarray:
        """Remove background using white-pixel threshold.

        Args:
            segment_bgr: Input segment in BGR format.

        Returns:
            4-channel BGRA image with transparent background.
        """
        return self._processor.remove_bg_threshold(segment_bgr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_processors.py::TestSegmentHandler -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_processors.py src/digitex/core/processors/image.py
git commit -m "feat: add background removal methods to SegmentHandler"
```

---

### Task 4: Update PageExtractor to handle new preprocess modes

**Files:**
- Modify: `src/digitex/extractors/page_extractor.py` (method `_crop_and_save`, lines 69-84)
- Test: `tests/test_processors.py` (integration tests)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_processors.py`:

```python
def test_crop_and_save_grabcut_forces_png(tmp_path: Path) -> None:
    """Test that grabcut preprocess mode saves as PNG regardless of image_format."""
    from unittest.mock import patch

    import cv2
    from PIL import Image as PILImage

    extractor = PageExtractor(
        model_path=Path("dummy.pt"),
        render_scale=2,
        image_format="jpg",
        preprocess="grabcut",
    )

    image = PILImage.new("RGB", (200, 200), color="white")
    polygon = [(10, 10), (190, 10), (190, 190), (10, 190)]
    output_path = tmp_path / "output.png"

    with patch.object(extractor._segment_handler, "remove_bg_grabcut") as mock_bg:
        mock_bg.return_value = np.ones((200, 200, 4), dtype=np.uint8) * 255
        extractor._crop_and_save(image, polygon, output_path)
        mock_bg.assert_called_once()

    assert output_path.exists()
    assert output_path.suffix == ".png"

    saved = PILImage.open(output_path)
    assert saved.mode == "RGBA"


def test_crop_and_save_threshold_forces_png(tmp_path: Path) -> None:
    """Test that threshold preprocess mode saves as PNG regardless of image_format."""
    from unittest.mock import patch

    import cv2
    from PIL import Image as PILImage

    extractor = PageExtractor(
        model_path=Path("dummy.pt"),
        render_scale=2,
        image_format="jpg",
        preprocess="threshold",
    )

    image = PILImage.new("RGB", (200, 200), color="white")
    polygon = [(10, 10), (190, 10), (190, 190), (10, 190)]
    output_path = tmp_path / "output.png"

    with patch.object(extractor._segment_handler, "remove_bg_threshold") as mock_bg:
        mock_bg.return_value = np.ones((200, 200, 4), dtype=np.uint8) * 255
        extractor._crop_and_save(image, polygon, output_path)
        mock_bg.assert_called_once()

    assert output_path.exists()
    assert output_path.suffix == ".png"

    saved = PILImage.open(output_path)
    assert saved.mode == "RGBA"
```

Add required imports at the top of `tests/test_processors.py`:

```python
from digitex.extractors.page_extractor import PageExtractor
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_processors.py -k "grabcut_forces_png or threshold_forces_png" -v`
Expected: FAIL — the existing `_crop_and_save` does not handle `grabcut`/`threshold` and will fall through to `binarize`.

- [ ] **Step 3: Update `_crop_and_save` implementation**

Replace the body of `_crop_and_save` in `src/digitex/extractors/page_extractor.py` (lines 69-84):

```python
    def _crop_and_save(
        self,
        image: Image.Image,
        polygon: list[tuple[int, int]],
        output_path: Path,
    ) -> None:
        cropped = self._image_cropper.cut_out_image_by_polygon(image, polygon)
        if self.preprocess:
            cropped_arr = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
            if self.preprocess == "enhance":
                processed = self._segment_handler.enhance(cropped_arr)
                cropped = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB))
            elif self.preprocess == "grabcut":
                processed = self._segment_handler.remove_bg_grabcut(cropped_arr)
                cropped = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGRA2RGBA))
                output_path = output_path.with_suffix(".png")
            elif self.preprocess == "threshold":
                processed = self._segment_handler.remove_bg_threshold(cropped_arr)
                cropped = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGRA2RGBA))
                output_path = output_path.with_suffix(".png")
            else:
                processed = self._segment_handler.binarize(cropped_arr)
                cropped = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_processors.py -k "grabcut_forces_png or threshold_forces_png" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/digitex/extractors/page_extractor.py tests/test_processors.py
git commit -m "feat: handle grabcut and threshold preprocess modes in PageExtractor"
```

---

### Task 5: Update CLI help text

**Files:**
- Modify: `extraction/run.py` (line 22)

- [ ] **Step 1: Update help text**

Replace line 22 in `extraction/run.py`:

From:
```python
        preprocess: Preprocessing mode: None, "enhance", or "binarize".
```

To:
```python
        preprocess: Preprocessing mode: None, "enhance", "binarize", "grabcut", or "threshold".
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add extraction/run.py
git commit -m "docs: update preprocess help text with grabcut and threshold modes"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 2: Run type checker**

Run: `uvx ty check src/digitex/core/processors/image.py src/digitex/extractors/page_extractor.py`
Expected: Zero errors
