import json
import logging
from pathlib import Path

import albumentations as A
import cv2
import doxapy
import numpy as np
import yaml
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_LOWER_BLUE = np.array([70, 30, 30])
DEFAULT_UPPER_BLUE = np.array([130, 255, 255])
DEFAULT_BIN_WINDOW = 30
DEFAULT_BIN_K = 0.16
DEFAULT_MAX_HEIGHT = 2000
DEFAULT_BORDER_MULTIPLIER = 5
DEFAULT_CROP_OFFSET = 0.025


class ImageProcessor:
    """Processor for image enhancement and transformation operations.

    This class provides methods for various image processing tasks including
    color removal, binarization, resizing, and illumination adjustments.
    """

    def __init__(self) -> None:
        """Initialize the ImageProcessor with default parameters."""
        self.scan_types = ["bw", "gray", "color"]
        self.lower_blue = DEFAULT_LOWER_BLUE
        self.upper_blue = DEFAULT_UPPER_BLUE
        self.bin_params = {"window": DEFAULT_BIN_WINDOW, "k": DEFAULT_BIN_K}

    def remove_color(self, img: np.ndarray) -> np.ndarray:
        """Remove blue color regions from an image using inpainting.

        Args:
            img: Input image in BGR format.

        Returns:
            Image with blue color regions removed.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        kernel = np.ones((DEFAULT_BORDER_MULTIPLIER, DEFAULT_BORDER_MULTIPLIER), np.uint8)
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

        wan = doxapy.Binarization(doxapy.Binarization.Algorithms.WAN)
        wan.initialize(gray)
        wan.to_binary(bin_img, self.bin_params)

        img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

        return img

    def resize_image(
        self,
        img: np.ndarray,
        max_height: int = DEFAULT_MAX_HEIGHT,
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

    def process(
        self,
        image: Image.Image,
        scan_type: str,
        resize: bool = False,
        remove_ink: bool = False,
        illuminate: bool = False,
        binarize: bool = False,
    ) -> Image.Image:
        """Apply a series of image processing operations.

        Args:
            image: Input PIL Image.
            scan_type: Type of scan ('bw', 'gray', or 'color').
            resize: Whether to resize the image.
            remove_ink: Whether to remove blue ink marks.
            illuminate: Whether to adjust image luminance.
            binarize: Whether to binarize the image.

        Returns:
            Processed PIL Image.

        Raises:
            ValueError: If scan_type is invalid.
        """
        if scan_type not in self.scan_types:
            raise ValueError(f"Scan type must be one of {self.scan_types}")

        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if resize:
            img = self.resize_image(img)

        if remove_ink:
            img = self.remove_color(img)

        if illuminate:
            img = self.illuminate_image(img)

        if binarize and scan_type != "bw":
            img = self.binarize_image(img)

        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class FileProcessor:
    """Processor for file I/O operations supporting various formats."""

    @staticmethod
    def read_txt(txt_path: str | Path) -> list[str]:
        """Read lines from a text file.

        Args:
            txt_path: Path to the text file.

        Returns:
            List of lines from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read.
        """
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            content = txt_file.readlines()

        return content

    @staticmethod
    def write_txt(txt_path: str | Path, lines: list[str]) -> None:
        """Write lines to a text file.

        Args:
            txt_path: Path to the output text file.
            lines: List of lines to write.

        Raises:
            IOError: If the file cannot be written.
        """
        with open(txt_path, 'w', encoding="utf-8") as txt_file:
            txt_file.writelines(lines)

    @staticmethod
    def read_json(json_path: str | Path) -> dict:
        """Read JSON data from a file.

        Args:
            json_path: Path to the JSON file.

        Returns:
            Dictionary containing the JSON data.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_dict = json.load(json_file)

        return json_dict

    @staticmethod
    def write_json(
        json_dict: dict,
        json_path: str | Path,
        indent: int = 4,
    ) -> None:
        """Write data to a JSON file.

        Args:
            json_dict: Dictionary to write as JSON.
            json_path: Path to the output JSON file.
            indent: Number of spaces for indentation.

        Raises:
            IOError: If the file cannot be written.
            TypeError: If the data is not JSON serializable.
        """
        with open(json_path, 'w', encoding="utf-8") as json_file:
            json.dump(json_dict, json_file, indent=indent, ensure_ascii=False)

    @staticmethod
    def write_yaml(
        yaml_path: str | Path,
        data: dict,
        comment: str | None = None,
    ) -> None:
        """Write data to a YAML file.

        Args:
            yaml_path: Path to the output YAML file.
            data: Dictionary to write as YAML.
            comment: Optional comment to add at the top of the file.

        Raises:
            IOError: If the file cannot be written.
        """
        with open(yaml_path, 'w', encoding="utf-8") as yaml_file:
            if comment:
                yaml_file.write(comment)
            yaml.dump(data, yaml_file, default_flow_style=False, allow_unicode=True)
