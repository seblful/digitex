from PIL import Image

import numpy as np
import cv2

import doxapy


class ImageProcessor:
    def __init__(self) -> None:
        # Scan
        self.scan_types = ["bw", "gray", "color"]

        # Blue remove range
        self.lower_blue = np.array([70, 30, 30])
        self.upper_blue = np.array([130, 255, 255])

        # Binarization
        self.bin_params = {"window": 25, "k": 0.16}

    def remove_color(self,
                     img: np.ndarray) -> np.ndarray:
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create a mask for blue color
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        # Dilate the mask to cover entire pen marks
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Inpaint the masked region
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        return img

    def binarize_image(self,
                       img: np.ndarray) -> np.ndarray:
        # Convert image to gray
        if len(img.shape) != 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Create empty binary image
        bin_img = np.empty(gray.shape, gray.dtype)

        # Convert the image to binary
        wan = doxapy.Binarization(doxapy.Binarization.Algorithms.WAN)
        wan.initialize(gray)
        wan.to_binary(bin_img, self.bin_params)

        # Convert image back to 3d
        img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

        return img

    def resize_image(self,
                     img: np.ndarray,
                     max_height: int = 2000) -> np.ndarray:

        height, width = img.shape[:2]

        # Check if the height is greater than the specified max height
        if height > max_height:
            # Calculate the aspect ratio
            aspect_ratio = width / height
            # Calculate the new dimensions
            new_height = max_height
            new_width = int(new_height * aspect_ratio)

            # Resize the image
            img = cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return img

    def process(self,
                image: Image.Image,
                scan_type: str,
                resize: bool = True,
                remove_ink: bool = True,
                binarize: bool = True) -> Image.Image:
        # Check scan type
        assert scan_type in self.scan_types, f"Scan type should be in one of {
            str(self.scan_types)}"

        # Convert image to array
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Resize image
        if resize is True:
            img = self.resize_image(img)

        # Remove ink
        if remove_ink is True:
            img = self.remove_color(img)

        # Binarize image
        if binarize is True and scan_type != "bw":
            img = self.binarize_image(img)

        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
