from PIL import Image

import numpy as np
import cv2


class ImageProcessor:
    def __init__(self,
                 fix_luminance: bool = True,
                 remove_ink: bool = True,
                 binarize: bool = False,
                 blur: bool = False) -> None:
        # Parameters
        self.fix_luminance = fix_luminance
        self.remove_ink = remove_ink
        self.binarize = binarize
        self.blur = blur

        # Blue remove range
        self.lower_blue = np.array([115, 150, 70])
        self.upper_blue = np.array([130, 255, 255])

    def remove_color(self,
                     img: np.array) -> None:
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

    def change_luminance(self,
                         img: np.array) -> None:
        # Calculate average contrast and brightness
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray_image)
        brightness = np.mean(gray_image)

        # Find alpha and beta
        alpha = round(np.log10(contrast) / 1.5, 3)
        beta = round(255 - brightness, 3)

        # # Print alpha and beta
        # print(f"Contrast: {contrast}, brightness: {brightness}")
        # print(f"Alpha: {alpha}, beta: {beta}")

        # Change luminance
        img = cv2.convertScaleAbs(img,
                                  alpha=alpha,
                                  beta=beta)

        return img

    def clean_image(self,
                    image: Image.Image) -> Image.Image:
        # Convert image to array
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if self.fix_luminance is True:
            img = self.change_luminance(img=img)

        # Binarize image
        if self.binarize is True:
            _, img = cv2.threshold(
                img, 150, 255, cv2.THRESH_BINARY)

        # Blur
        if self.blur is True:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        # Remove ink
        if self.remove_ink is True:
            img = self.remove_color(img=img)

        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
