import os

from modules.data_creator import DataCreator

# Paths
HOME = os.getcwd()

TESTING_DIR = os.path.dirname(os.path.dirname(HOME))
IMAGES_DIR = os.path.join(TESTING_DIR, "raw-data", "images")
RAW_DIR = os.path.join(TESTING_DIR, "raw-data", "new")
TRAIN_DIR = os.path.join(HOME, "data", "train-data")


def main() -> None:
    # Create DataCreator instance
    data_creator = DataCreator()

    # # Creat pdfs from images
    # for image_dir in os.listdir(IMAGES_DIR):
    #     image_dir = os.path.join(IMAGES_DIR, image_dir)
    #     data_creator.create_pdf_from_images(image_dir=image_dir,
    #                                         raw_dir=RAW_DIR)

    # Create data
    data_creator.create_yolo_train_data(raw_dir=RAW_DIR,
                                        train_dir=TRAIN_DIR,
                                        scan_type="color",
                                        num_images=100)


if __name__ == "__main__":
    main()
