import os

from modules.data_creator import DataCreator

# Paths
HOME = os.getcwd()

TESTING_DIR = os.path.dirname(os.path.dirname(HOME))
IMAGES_DIR = os.path.join(TESTING_DIR, "raw-data", "images")
RAW_DATA_DIR = os.path.join(TESTING_DIR, "raw-data", "new")
TRAIN_PAGE_DIR = os.path.join(HOME, "data", "page", "train-data")
TRAIN_QUESTION_DIR = os.path.join(HOME, "data", "question", "train-data")

PAGE_RAW_DIR = os.path.join(HOME, "data", "page", "raw-data")
YOLO_PAGE_PATH = os.path.join("models", "yolov11", "page.pt")


def main() -> None:
    # Create DataCreator instance
    data_creator = DataCreator()

    # # Creat pdfs from images
    # for image_dir in os.listdir(IMAGES_DIR):
    #     image_dir = os.path.join(IMAGES_DIR, image_dir)
    #     data_creator.create_pdf_from_images(image_dir=image_dir,
    #                                         raw_dir=RAW_DATA_DIR)

    # # Create data for page
    # data_creator.create_yolo_train_data(raw_dir=RAW_DATA_DIR,
    #                                     train_dir=TRAIN_PAGE_DIR,
    #                                     scan_type="color",
    #                                     num_images=100)

    # # Create data for question from annotations
    # data_creator.create_question_train_data_raw(page_raw_dir=PAGE_RAW_DIR,
    #                                             train_dir=TRAIN_QUESTION_DIR,
    #                                             num_images=100)

    # Create data for question from YOLO predictions
    data_creator.create_question_train_data_pred(raw_dir=RAW_DATA_DIR,
                                                 train_dir=TRAIN_QUESTION_DIR,
                                                 yolo_model_path=YOLO_PAGE_PATH,
                                                 num_images=100)


if __name__ == "__main__":
    main()
