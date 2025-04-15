from PIL import Image

import numpy as np
import cv2
import doxapy


class ImgProcessor:
    # Blue remove
    LOWER_BLUE = np.array([70, 30, 30])
    UPPER_BLUE = np.array([130, 255, 255])
    KERNEL_SIZE = (5, 5)

    # Binarization
    BIN_PARAMS = {"window": 30, "k": 0.16}

    @staticmethod
    def image2img(image: Image) -> np.ndarray:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        return img

    @staticmethod
    def img2image(img: np.ndarray) -> Image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)

        return image

    @staticmethod
    def resize_image(
        img: np.ndarray, target_width: int, target_height: int
    ) -> np.ndarray:
        img_height, img_width = img.shape[:2]

        # Calculate scaling factor while maintaining aspect ratio
        width_ratio = target_width / img_width
        height_ratio = target_height / img_height
        scale_factor = min(width_ratio, height_ratio)

        # Calculate new dimensions
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)

        # Resize the image
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
        )

        return resized_img

    @staticmethod
    def illuminate_image(
        img: np.ndarray, alpha: float = 1.1, beta: int = 1
    ) -> np.ndarray:
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img

    @staticmethod
    def binarize_image(img: np.ndarray) -> np.ndarray:
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
        wan.to_binary(bin_img, ImgProcessor.BIN_PARAMS)

        # Convert image back to 3d
        img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

        return img

    @staticmethod
    def remove_blue(img: np.ndarray) -> np.ndarray:
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create a mask for blue color
        mask = cv2.inRange(hsv, ImgProcessor.LOWER_BLUE, ImgProcessor.UPPER_BLUE)

        # Dilate the mask to cover entire pen marks
        kernel = np.ones(ImgProcessor.KERNEL_SIZE, np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Inpaint the masked region
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        return img
