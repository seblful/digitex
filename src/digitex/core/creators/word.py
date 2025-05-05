import os
from digitex.settings import settings
from .base import BaseDataCreator
from ..predictors.segmentation import YOLO_SegmentationPredictor
from ..predictors.detection import DB_RepVitDetectionPredictor


class WordDataCreator(BaseDataCreator):
    def extract(self, parts_raw_dir: str, train_dir: str, num_images: int) -> None:
        images_dir = os.path.join(parts_raw_dir, "images")
        labels_dir = os.path.join(parts_raw_dir, "labels")
        classes_path = os.path.join(parts_raw_dir, "classes.txt")
        classes_dict = self._read_classes(classes_path)
        target_classes = ["text"]
        images_listdir = os.listdir(images_dir)
        num_saved = 0

        while num_images != num_saved:
            rand_image, rand_image_name = self._get_listdir_random_image(
                images_listdir, images_dir
            )
            rand_points_idx, rand_points = self._get_points(
                image_name=rand_image_name,
                labels_dir=labels_dir,
                classes_dict=classes_dict,
                target_classes=target_classes,
            )
            rand_polygon = self._convert_points_to_polygon(
                points=rand_points,
                image_width=rand_image.width,
                image_height=rand_image.height,
            )

            cropped_image = self._crop_image(image=rand_image, polygon=rand_polygon)
            num_saved = self._save_image(
                rand_points_idx,
                output_dir=train_dir,
                image=cropped_image,
                image_name=rand_image_name,
                num_saved=num_saved,
                num_images=num_images,
            )

    def predict(
        self,
        pdf_dir: str,
        train_dir: str,
        yolo_page_model_path: str,
        yolo_question_model_path: str,
        db_repvit_word_model_path: str,
        db_repvit_word_config_path: str,
        num_images: int,
    ) -> None:
        yolo_page_predictor = YOLO_SegmentationPredictor(
            yolo_page_model_path, device=settings.DEVICE
        )
        yolo_question_predictor = YOLO_SegmentationPredictor(
            yolo_question_model_path, device=settings.DEVICE
        )
        db_repvit_word_predictor = DB_RepVitDetectionPredictor(
            db_repvit_word_model_path,
            db_repvit_word_config_path,
            device=settings.DEVICE,
        )
        pdf_listdir = [pdf for pdf in os.listdir(pdf_dir) if pdf.endswith("pdf")]
        parts_target_classes = ["answer", "number", "option", "question", "spec"]
        num_saved = 0

        while num_images != num_saved:
            page_rand_image, page_rand_image_name, rand_page_idx = (
                self._get_pdf_random_image(pdf_listdir, pdf_dir)
            )
            page_rand_image = self._process_image(image=page_rand_image)
            page_pred_result = yolo_page_predictor(page_rand_image)
            page_points_dict = page_pred_result.id2polygons
            question_rand_idx, question_rand_polygon = (
                self.label_handler._get_random_points(
                    classes_dict=page_pred_result.id2label,
                    points_dict=page_points_dict,
                    target_classes=["question"],
                )
            )
            question_rand_image = self._cut_out_image(
                image=page_rand_image, polygon=question_rand_polygon
            )
            question_pred_result = yolo_question_predictor(question_rand_image)
            question_points_dict = question_pred_result.id2polygons
            part_rand_idx, part_rand_polygon = self.label_handler._get_random_points(
                classes_dict=question_pred_result.id2label,
                points_dict=question_points_dict,
                target_classes=parts_target_classes,
            )
            part_rand_image = self._cut_out_image(
                image=question_rand_image, polygon=part_rand_polygon
            )
            word_pred_result = db_repvit_word_predictor(part_rand_image)

            if not word_pred_result:
                continue

            word_points_dict = word_pred_result.id2polygons
            word_rand_idx, word_rand_polygon = self.label_handler._get_random_points(
                classes_dict=word_pred_result.id2label,
                points_dict=word_points_dict,
                target_classes=["text"],
            )

            word_rand_image = self._crop_image(
                image=part_rand_image, polygon=word_rand_polygon
            )
            num_saved = self._save_image(
                rand_page_idx,
                question_rand_idx,
                part_rand_idx,
                word_rand_idx,
                output_dir=train_dir,
                image=word_rand_image,
                image_name=page_rand_image_name,
                num_saved=num_saved,
                num_images=num_images,
            )
