import os
from PIL import Image, ImageDraw
from modules.handlers import PDFHandler, ImageHandler
from modules.processors import FileProcessor
from modules.predictors.segmentation import YOLO_SegmentationPredictor


class PDFManager:
    def __init__(self, pdf_handler: PDFHandler, file_processor: FileProcessor, inputs_dir: str) -> None:
        self.pdf_handler = pdf_handler
        self.file_processor = file_processor
        self.ckpt_path = os.path.join(inputs_dir, "checkpoints.json")
        self.current_pdf_path = None
        self.current_pdf_obj = None
        self.current_page = 0
        self.page_count = 0

    def open_pdf(self, pdf_path: str) -> None:
        self.current_pdf_path = pdf_path
        self.current_pdf_obj = self.pdf_handler.open_pdf(pdf_path)
        self.current_page = 0
        self.page_count = len(self.current_pdf_obj)

    def save_checkpoint(self) -> None:
        checkpoint = {"pdf_path": self.current_pdf_path,
                      "page": self.current_page}
        self.file_processor.write_json(checkpoint, self.ckpt_path)

    def load_checkpoint(self) -> dict:
        return self.file_processor.read_json(self.ckpt_path)


class ImageManager:
    def __init__(self, image_handler: ImageHandler, base_image_dimensions: tuple) -> None:
        self.image_handler = image_handler
        self.pdf_handler = PDFHandler()
        self.base_image_dimensions = base_image_dimensions
        self.original_image = None
        self.base_image = None
        self.resized_image = None
        self.tk_image = None

    def load_page_image(self, pdf_page) -> None:
        self.original_image = self.pdf_handler.get_page_image(pdf_page)
        self.base_image = self.image_handler.resize_image(
            self.original_image, *self.base_image_dimensions
        )

    def resize_image(self, zoom_level: float, canvas_width: int, canvas_height: int) -> Image.Image:
        if zoom_level == 1.0:
            return self.image_handler.resize_image(self.base_image, canvas_width, canvas_height)
        else:
            width = int(self.base_image.width * zoom_level)
            height = int(self.base_image.height * zoom_level)
            return self.base_image.resize((width, height), Image.Resampling.LANCZOS)


class PredictionManager:
    def __init__(self, cfg: dict, image_handler: ImageHandler) -> None:
        self._load_models(cfg)

        self.image_handler = image_handler
        self.colors = self._initialize_colors()
        self.question_images = []

    def _load_models(self, cfg: dict[str, str]) -> None:
        self.page_predictor = YOLO_SegmentationPredictor(
            cfg["model_path"]["page"])
        self.question_predictor = YOLO_SegmentationPredictor(
            cfg["model_path"]["question"])

    @staticmethod
    def _initialize_colors() -> dict:
        return {
            0: (255, 0, 0, 128),
            1: (0, 255, 0, 128),
            2: (0, 0, 255, 128),
            3: (255, 255, 0, 128),
            4: (255, 0, 255, 128),
            5: (0, 255, 255, 128),
            6: (128, 0, 128, 128),
            7: (255, 165, 0, 128),
        }

    def run_ml(self, original_image: Image.Image) -> tuple:
        page_predictions = self.page_predictor.predict(original_image)
        drawn_image = self._draw_polygons(
            original_image, page_predictions.id2polygons)
        self.question_images = [
            self.image_handler.crop_image(original_image, polygon)
            for cls, polygons in page_predictions.id2polygons.items()
            if page_predictions.id2label[cls] == "question"
            for polygon in polygons
        ]
        return drawn_image, len(self.question_images)

    def _draw_polygons(self, image: Image.Image, id2polygons: dict) -> Image.Image:
        drawn_image = image.copy()
        draw = ImageDraw.Draw(drawn_image, "RGBA")
        for cls, polygons in id2polygons.items():
            for polygon in polygons:
                draw.polygon(polygon, fill=self.colors[cls], outline="black")
        return drawn_image
