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
    def get_perspective_matrix(
        self, pts: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        # Create destination points
        dst_points = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(pts, dst_points)
        return M

    def perspective_transform(
        self, polygon: list[tuple[int, int]], persp_M: np.ndarray
    ) -> np.ndarray:
        poly_np = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
        warped_poly = cv2.perspectiveTransform(poly_np, persp_M).astype(np.int32)
        return warped_poly

    def warp_perspective(
        self, img: np.ndarray, persp_M: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        warped_img = cv2.warpPerspective(img, persp_M, (width, height))
        return warped_img

    def get_quadrilateral_size(self, pts: np.ndarray) -> tuple[int, int]:
        width_a = np.linalg.norm(pts[0] - pts[1])
        width_b = np.linalg.norm(pts[2] - pts[3])
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(pts[1] - pts[2])
        height_b = np.linalg.norm(pts[3] - pts[0])
        max_height = max(int(height_a), int(height_b))

        return max_width, max_height

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def polygon_to_quadrilateral(
        self,
        polygon: list[tuple[int, int]],
        max_angle: float = 4.0,
    ) -> np.ndarray:
        pts = np.array(polygon, dtype=np.int32)

        # Compute the minimum area rectangle enclosing the polygon
        rect = cv2.minAreaRect(pts)

        # Check the angle of the rectangle
        # If the angle is too large, use cv2.boundingRect
        angle = rect[2]
        angle_delta = abs(min(angle - 0, 90 - angle))
        if angle_delta > max_angle:
            x, y, w, h = cv2.boundingRect(pts)
            bbox = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
            )
        else:
            bbox = cv2.boxPoints(rect)

        # Order the points for perspective transform
        bbox = self.order_points(bbox)

        return bbox

    def paste_img_on_bg(
        self, img: np.ndarray, pts: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        # Create mask for the polygon in the image
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Apply mask to image
        result = cv2.bitwise_and(img, img, mask=mask)

        # Paste on white background to remove black outside polygon
        white_bg = np.ones_like(result) * 255
        mask_inv = cv2.bitwise_not(mask)
        white_masked = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
        bg_img = cv2.add(result, white_masked)

        return bg_img

    def crop_img_by_box(
        self, img: np.ndarray, box: tuple[int, int, int, int], angle: int = 0
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
        persp_M = self.get_perspective_matrix(rotated_pts, width, height)
        cropped_img = self.warp_perspective(img, persp_M, width, height)

        return cropped_img

    def crop_img_by_polygon(
        self, img: np.ndarray, polygon: list[tuple[int, int]]
    ) -> np.ndarray:
        # Convert polygon to numpy quadrilateral
        if len(polygon) == 4:
            pts = np.array(polygon, dtype=np.float32)
        elif len(polygon) > 4:
            pts = self.polygon_to_quadrilateral(polygon)
        else:
            raise ValueError("Polygon must have 4 or more than 4 points.")

        # Determine the width and height of the output image
        width, height = self.get_quadrilateral_size(pts)

        # Crop the image
        persp_M = self.get_perspective_matrix(pts, width, height)
        cropped_img = self.warp_perspective(img, persp_M, int(width), int(height))

        return cropped_img

    def cut_out_img_by_polygon(
        self, img: np.ndarray, polygon: list[tuple[int, int]]
    ) -> np.ndarray:
        # Convert polygon to quadrilateral (minimum area rectangle enclosing the polygon)
        pts = self.polygon_to_quadrilateral(polygon)

        # Determine the width and height of the quadrilateral
        width, height = self.get_quadrilateral_size(pts)

        # Compute the perspective transformation matrix for the quadrilateral
        persp_M = self.get_perspective_matrix(pts, width, height)

        # Apply the perspective transformation to warp the image
        warped_img = self.warp_perspective(img, persp_M, width, height)

        # Transform the original polygon points using the perspective matrix
        tr_pts = self.perspective_transform(polygon, persp_M)

        # Paste the warped image onto a white background using the transformed polygon points
        bg_img = self.paste_img_on_bg(warped_img, tr_pts, width, height)

        return bg_img
