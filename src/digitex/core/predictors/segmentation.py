from PIL import Image

import numpy as np
import torch
import cv2
import yaml
import os

from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers.modeling_outputs import SemanticSegmenterOutput

from .abstract_predictor import Predictor
from .prediction_result import SegmentationPredictionResult


class YOLO_SegmentationPredictor(Predictor):
    def __init__(self, model_path: str, device: torch.device) -> None:
        self.model_path = model_path
        self.device = device
        self.__model = None

    @property
    def model(self) -> YOLO:
        if self.__model is None:
            self.__model = YOLO(self.model_path, verbose=False)

        return self.__model

    def preprocess_image(self, image: Image) -> np.ndarray:
        img = np.array(image)
        return img

    def create_result(
        self, preds: list[dict], img_width: int, img_height: int
    ) -> SegmentationPredictionResult:
        ids = []
        polygons = []

        # Process bboxes, ids and append to lists
        for box, polygon in zip(preds[0].boxes, preds[0].masks.xyn):
            polygon = polygon * np.array([img_width, img_height])
            polygon = polygon.astype(np.int32)
            polygon = polygon.tolist()
            polygon = [tuple(points) for points in polygon]

            idx = int(box.cls.item())

            ids.append(idx)
            polygons.append(polygon)

        # Create result
        result = SegmentationPredictionResult(
            ids=ids, polygons=polygons, id2label=self.model.names
        )

        return result

    def predict(self, image: Image.Image) -> SegmentationPredictionResult:
        img = self.preprocess_image(image)
        img_height, img_width, _ = img.shape
        preds = self.model.predict(img, verbose=False)
        result = self.create_result(preds, img_width, img_height)

        return result


class SegformerSegmentationPredictor(Predictor):
    def __init__(self, config_path: str, model_path: str, device: torch.device) -> None:
        self.config_path = config_path
        self.model_path = model_path
        self.device = device

        self.__config = None
        self.__model = None
        self.__image_processor = None

    @property
    def config(self) -> dict:
        if self.__config is None:
            with open(self.config_path, "r") as f:
                self.__config = yaml.safe_load(f)
        return self.__config

    @property
    def model(self) -> SegformerForSemanticSegmentation:
        if self.__model is None:
            if os.path.isdir(self.model_path):
                # Load as safetensors (directory)
                self.__model = SegformerForSemanticSegmentation.from_pretrained(
                    self.model_path, ignore_mismatched_sizes=False
                )
            elif self.model_path.endswith(".pth"):
                # Load as .pth file
                self.__model = SegformerForSemanticSegmentation.from_pretrained(
                    self.config["model"]["model_name"],
                    num_labels=self.config["dataset"]["num_classes"],
                    id2label=self.config["model"]["id2label"],
                    label2id=self.config["model"]["label2id"],
                    ignore_mismatched_sizes=True,
                )
                checkpoint = torch.load(
                    self.model_path, map_location=self.device, weights_only=False
                )
                self.__model.load_state_dict(checkpoint["model_state_dict"])
            else:
                raise ValueError(
                    f"Unsupported model path format: {self.model_path}. Expected directory or .pth file."
                )

            self.__model.to(self.device)
            self.__model.eval()
        return self.__model

    @property
    def image_processor(self) -> SegformerImageProcessor:
        if self.__image_processor is None:
            self.__image_processor = SegformerImageProcessor.from_pretrained(
                self.config["model"]["model_name"]
            )
        return self.__image_processor

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        image_size = self.config["dataset"]["image_size"]
        height, width = image_size[0], image_size[1]

        inputs = self.image_processor(
            image,
            return_tensors="pt",
            do_resize=True,
            size={"height": height, "width": width},
            do_normalize=True,
        )
        return inputs["pixel_values"].to(self.device)

    def _mask_to_polygons(
        self, mask: np.ndarray, eps_factor: float = 0.003, min_points: int = 5
    ) -> list[list[tuple[int, int]]]:
        polygons = []

        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Simplify contour to reduce number of points
            epsilon = eps_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Convert to list of tuples
            polygon = [(int(point[0][0]), int(point[0][1])) for point in approx]

            # Only keep polygons with minimum required points
            if len(polygon) >= min_points:
                polygons.append(polygon)

        return polygons

    def create_result(
        self,
        preds: SemanticSegmenterOutput,
        image_width: int,
        image_height: int,
    ) -> SegmentationPredictionResult:
        # Post-process outputs to get binary maps
        segmentation_maps = self.image_processor.post_process_semantic_segmentation(
            preds, target_sizes=[(image_height, image_width)]
        )
        segmentation_map = segmentation_maps[0].cpu().numpy()  # Shape: (H, W)
        class_mask = (segmentation_map == 1).astype(np.uint8)

        # Resize segmentation map to original image size
        resized_mask = cv2.resize(
            class_mask,
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        )

        # Convert mask to polygons
        polygons = self._mask_to_polygons(resized_mask)
        ids = [0] * len(polygons)

        # Create result
        result = SegmentationPredictionResult(
            ids=ids,
            polygons=polygons,
            id2label={0: self.config["model"]["id2label"][1]},
        )

        return result

    def predict(self, image: Image.Image) -> SegmentationPredictionResult:
        # Get original image dimensions
        image_width, image_height = image.size

        # Preprocess image
        img = self.preprocess_image(image)

        # Run inference
        with torch.no_grad():
            preds = self.model(img)

        # Create result
        result = self.create_result(preds, image_width, image_height)

        return result
