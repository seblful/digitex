"""Image processing utilities."""

import logging

import cv2
import doxapy  # type: ignore[import-untyped]
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Processor for image enhancement and transformation operations.

    This class provides methods for various image processing tasks including
    color removal, binarization, resizing, and illumination adjustments.
    """

    def __init__(
        self,
        lower_blue: np.ndarray | None = None,
        upper_blue: np.ndarray | None = None,
        bin_window: int = 30,
        bin_k: float = 0.16,
        border_multiplier: int = 5,
    ) -> None:
        self.lower_blue = lower_blue if lower_blue is not None else np.array([70, 30, 30])
        self.upper_blue = upper_blue if upper_blue is not None else np.array([130, 255, 255])
        self.bin_params = {"window": bin_window, "k": bin_k}
        self.border_multiplier = border_multiplier

    def remove_color(self, img: np.ndarray) -> np.ndarray:
        """Remove blue color regions from an image using inpainting.

        Args:
            img: Input image in BGR format.

        Returns:
            Image with blue color regions removed.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        kernel = np.ones(
            (self.border_multiplier, self.border_multiplier), np.uint8
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        return img

    def illuminate_image(
        self,
        img: np.ndarray,
        alpha: float = 1.1,
        beta: float = 1,
    ) -> np.ndarray:
        """Adjust image luminance using linear transformation.

        Args:
            img: Input image in BGR format.
            alpha: Contrast control (1.0 means no change).
            beta: Brightness control (0 means no change).

        Returns:
            Illuminated image.
        """
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img

    def binarize_image(self, img: np.ndarray) -> np.ndarray:
        """Convert image to binary using the Wan binarization algorithm.

        Args:
            img: Input image in BGR or grayscale format.

        Returns:
            Binary image in BGR format.
        """
        if len(img.shape) != 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        bin_img = np.empty(gray.shape, gray.dtype)

        wan = doxapy.Binarization(doxapy.Binarization.Algorithms.WAN)  # ty: ignore[unresolved-attribute]
        wan.initialize(gray)
        wan.to_binary(bin_img, self.bin_params)

        img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

        return img

    def resize_image(
        self,
        img: np.ndarray,
        max_height: int = 2000,
    ) -> np.ndarray:
        """Resize image to fit within maximum height while maintaining aspect ratio.

        Args:
            img: Input image.
            max_height: Maximum allowed height in pixels.

        Returns:
            Resized image if height exceeds max_height, otherwise original image.
        """
        height, width = img.shape[:2]

        if height > max_height:
            aspect_ratio = width / height
            new_height = max_height
            new_width = int(new_height * aspect_ratio)

            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return img

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale.

        Args:
            img: Input image in BGR format.

        Returns:
            Grayscale image.
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def denoise(
        img: np.ndarray,
        d: int = 5,
        sigma_color: float = 75,
        sigma_space: float = 75,
    ) -> np.ndarray:
        """Apply bilateral filter for edge-preserving denoising.

        Args:
            img: Input grayscale image.
            d: Diameter of pixel neighborhood.
            sigma_color: Filter sigma in color space.
            sigma_space: Filter sigma in coordinate space.

        Returns:
            Denoised image.
        """
        return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    @staticmethod
    def apply_clahe(
        img: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            img: Input grayscale image.
            clip_limit: Threshold for contrast limiting.
            tile_size: Size of grid for histogram equalization.

        Returns:
            Contrast-enhanced image.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(img)

    @staticmethod
    def adjust_contrast(
        img: np.ndarray,
        alpha: float = 1.3,
        beta: float = -15,
    ) -> np.ndarray:
        """Adjust image contrast and brightness.

        Args:
            img: Input image.
            alpha: Contrast control (1.0 means no change).
            beta: Brightness control (0 means no change).

        Returns:
            Adjusted image.
        """
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    @staticmethod
    def whiten_background(img: np.ndarray, threshold: int = 200) -> np.ndarray:
        """Set bright pixels to white for clean background.

        Args:
            img: Input image (modified in place).
            threshold: Brightness threshold.

        Returns:
            Image with whitened background.
        """
        result = img.copy()
        result[result > threshold] = 255
        return result

    @staticmethod
    def apply_wan_binarization(
        img: np.ndarray,
        window: int | None = None,
        k: float = 0.2,
    ) -> np.ndarray:
        """Apply Wan adaptive binarization algorithm.

        Args:
            img: Input grayscale image.
            window: Window size for local thresholding. If None, auto-calculated.
            k: Sensitivity parameter.

        Returns:
            Binary image.
        """
        if window is None:
            min_dim = min(img.shape)
            window = max(15, min_dim // 20)
            if window % 2 == 0:
                window += 1

        bin_img = np.empty(img.shape, img.dtype)
        wan = doxapy.Binarization(doxapy.Binarization.Algorithms.WAN)  # ty: ignore[unresolved-attribute]
        wan.initialize(img)
        wan.to_binary(bin_img, {"window": window, "k": k})
        return bin_img

    @staticmethod
    def apply_morphology(img: np.ndarray, kernel_size: int = 2) -> np.ndarray:
        """Apply morphological open then close operations.

        Args:
            img: Input binary image.
            kernel_size: Size of structuring element.

        Returns:
            Morphologically processed image.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

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

        Raises:
            ValueError: If image is empty, has wrong dtype, wrong number of channels,
                        or if GrabCut algorithm fails.
        """
        if image_bgr.size == 0:
            raise ValueError("Image is empty")
        if image_bgr.dtype != np.uint8:
            raise ValueError(f"Image must have dtype uint8, got {image_bgr.dtype}")
        if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Image must have 3 channels, got shape {image_bgr.shape}")
        height, width = image_bgr.shape[:2]
        if height <= 2 or width <= 2:
            raise ValueError(f"Image must be larger than 2x2, got {width}x{height}")

        mask = np.zeros(image_bgr.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (1, 1, width - 2, height - 2)

        try:
            cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)
        except Exception as e:
            raise ValueError(f"GrabCut algorithm failed: {e}") from e

        binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        b, g, r = cv2.split(image_bgr)
        return cv2.merge([b, g, r, binary_mask])

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
        if image_bgr.size == 0:
            raise ValueError("Image is empty")
        if image_bgr.dtype != np.uint8:
            raise ValueError(f"Image must have dtype uint8, got {image_bgr.dtype}")
        if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Image must have 3 channels, got shape {image_bgr.shape}")
        if not isinstance(threshold, int) or not (0 <= threshold <= 255):
            raise ValueError(f"Threshold must be an integer in range 0-255, got {threshold}")

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        b, g, r = cv2.split(image_bgr)
        return cv2.merge([b, g, r, binary_mask])


class ImageCropper:
    """Processor for image cropping operations using perspective transformations."""

    @staticmethod
    def _get_perspective_matrix(
        pts: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        dst_points = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(pts, dst_points)

    @staticmethod
    def _perspective_transform(
        polygon: list[tuple[int, int]],
        persp_M: np.ndarray,
    ) -> np.ndarray:
        poly_np = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
        warped_poly = cv2.perspectiveTransform(poly_np, persp_M).astype(np.int32)
        return warped_poly

    @staticmethod
    def _warp_perspective(
        img: np.ndarray,
        persp_M: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        return cv2.warpPerspective(img, persp_M, (width, height))

    @staticmethod
    def _get_quadrilateral_size(pts: np.ndarray) -> tuple[int, int]:
        width_a = np.linalg.norm(pts[0] - pts[1])
        width_b = np.linalg.norm(pts[2] - pts[3])
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(pts[1] - pts[2])
        height_b = np.linalg.norm(pts[3] - pts[0])
        max_height = max(int(height_a), int(height_b))

        return max_width, max_height

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def _polygon_to_quadrilateral(
        polygon: list[tuple[int, int]],
        max_angle: float = 4.0,
    ) -> np.ndarray:
        pts = np.array(polygon, dtype=np.int32)

        rect = cv2.minAreaRect(pts)
        angle = rect[2]
        angle_delta = abs(min(angle - 0, 90 - angle))

        if angle_delta > max_angle:
            x, y, w, h = cv2.boundingRect(pts)
            bbox = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                dtype=np.float32,
            )
        else:
            bbox = cv2.boxPoints(rect)

        return ImageCropper._order_points(bbox)

    @staticmethod
    def _paste_on_white_background(
        img: np.ndarray,
        pts: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        result = cv2.bitwise_and(img, img, mask=mask)

        white_bg = np.ones_like(result) * 255
        mask_inv = cv2.bitwise_not(mask)
        white_masked = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
        return cv2.add(result, white_masked)

    def crop_image_by_polygon(
        self,
        image: Image.Image,
        polygon: list[tuple[int, int]],
    ) -> Image.Image:
        if len(polygon) < 4:
            raise ValueError("Polygon must have 4 or more points")

        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if len(polygon) == 4:
            pts = np.array(polygon, dtype=np.float32)
        else:
            pts = self._polygon_to_quadrilateral(polygon)

        width, height = self._get_quadrilateral_size(pts)
        persp_M = self._get_perspective_matrix(pts, width, height)
        cropped_img = self._warp_perspective(img, persp_M, int(width), int(height))

        return Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

    def cut_out_image_by_polygon(
        self,
        image: Image.Image,
        polygon: list[tuple[int, int]],
    ) -> Image.Image:
        if len(polygon) < 4:
            raise ValueError("Polygon must have 4 or more points")

        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        pts = self._polygon_to_quadrilateral(polygon)
        width, height = self._get_quadrilateral_size(pts)
        persp_M = self._get_perspective_matrix(pts, width, height)
        warped_img = self._warp_perspective(img, persp_M, width, height)

        tr_pts = self._perspective_transform(polygon, persp_M)
        bg_img = self._paste_on_white_background(warped_img, tr_pts, width, height)

        return Image.fromarray(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))


class SegmentHandler:
    """Handler for image segment enhancement and binarization.

    Uses ImageProcessor methods to process image segments with a shared
    preprocessing pipeline: remove_color → to_grayscale → denoise → apply_clahe.
    """

    def __init__(self, image_processor: ImageProcessor | None = None) -> None:
        """Initialize SegmentHandler.

        Args:
            image_processor: Optional ImageProcessor instance. If None, creates one.
        """
        self._processor = image_processor or ImageProcessor()

    def preprocess(self, segment_bgr: np.ndarray) -> np.ndarray:
        """Apply standard preprocessing pipeline for segments.

        Pipeline: remove_color → to_grayscale → denoise → apply_clahe

        Args:
            segment_bgr: Input segment in BGR format.

        Returns:
            Preprocessed grayscale image.
        """
        no_blue = self._processor.remove_color(segment_bgr)
        gray = self._processor.to_grayscale(no_blue)
        denoised = self._processor.denoise(gray)
        return self._processor.apply_clahe(denoised)

    def enhance(self, segment_bgr: np.ndarray) -> np.ndarray:
        """Enhance segment for better readability.

        Args:
            segment_bgr: Input segment in BGR format.

        Returns:
            Enhanced grayscale image with brightened background.
        """
        result = self.preprocess(segment_bgr)
        result = self._processor.adjust_contrast(result)
        return self._processor.whiten_background(result)

    def binarize(
        self,
        segment_bgr: np.ndarray,
        use_morphology: bool = False,
    ) -> np.ndarray:
        """Binarize segment using Wan algorithm.

        Args:
            segment_bgr: Input segment in BGR format.
            use_morphology: Whether to apply morphological operations.

        Returns:
            Binary image.
        """
        result = self.preprocess(segment_bgr)
        result = self._processor.apply_wan_binarization(result)
        if use_morphology:
            result = self._processor.apply_morphology(result)
        return result

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
