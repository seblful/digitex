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
    def resize_img(
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


class ImgCropper:
    @staticmethod
    def isolate_box(polygon: list[tuple[int, int]]) -> tuple[tuple, np.ndarray]:
        contours = np.array(polygon, dtype=np.int32)
        # Change to cv2 minarea rect, because rotated bbox
        box = cv2.boundingRect(contours)
        contours = contours - [box[0], box[1]]
        return box, contours

    @staticmethod
    def crop_to_mask(
        img: np.ndarray, box: tuple[int, int, int, int], contours: np.ndarray
    ) -> np.ndarray:
        # Crop the image to the box
        x, y, w, h = box
        cropped_img = img[y : y + h, x : x + w]

        # Create a mask and apply it to the cropped image
        mask = np.zeros(cropped_img.shape[:2], dtype=np.uint8)

        # TODO maybe delete this line
        cv2.drawContours(
            mask, [contours], -1, 255, thickness=cv2.FILLED, lineType=cv2.LINE_AA
        )
        result = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)
        return result

    @staticmethod
    def warp_perspective(
        img: np.ndarray, pts: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        # Create destination points
        dst_points = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(pts, dst_points)

        # Warp the image
        warped_img = cv2.warpPerspective(img, M, (width, height))

        return warped_img

    @staticmethod
    def crop_img_by_polygon(
        img: np.ndarray, polygon: list[tuple[int, int]]
    ) -> np.ndarray:
        # TODO convert polygon to xyxyxyxy if len > 8

        # Convert points to numpy array
        pts = np.array(polygon, dtype="float32")

        # Determine the width and height of the output image
        width = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
        height = max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))

        # Crop the image
        cropped_img = ImgCropper.warp_perspective(img, pts, int(width), int(height))

        return cropped_img

    @staticmethod
    def crop_img_by_box(
        img: np.ndarray, box: tuple[int, int, int, int], angle: int = 0
    ) -> np.ndarray:
        x, y, width, height = box

        # Get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((x, y), angle, 1.0)

        # Calculate the four corners of the rectangle and rotate them
        pts = np.array(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            dtype=np.float32,
        )
        rotated_pts = cv2.transform(np.array([pts]), rotation_matrix)[0]

        # Crop the image
        cropped_img = ImgCropper.warp_perspective(img, rotated_pts, width, height)

        return cropped_img

    @staticmethod
    def cut_out_img_by_polygon(
        img: np.ndarray, polygon: list[tuple[int, int]]
    ) -> np.ndarray:
        bbox, contours = ImgCropper.isolate_box(polygon)
        cropped_img = ImgCropper.crop_to_mask(img, bbox, contours)
        return cropped_img

    @staticmethod
    def paste_img_on_bg(img: np.ndarray, offset: float = 0.025) -> np.ndarray:
        # Create a white background
        bg = np.full_like(img, 255, dtype=np.uint8)

        # Create a mask from the image
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

        # Combine the image with the background using the mask
        bg_img = cv2.add(img, cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask)))

        # Add a border around the result
        border = int(img.shape[0] * offset)
        bg_img = cv2.copyMakeBorder(
            bg_img,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
        return bg_img
