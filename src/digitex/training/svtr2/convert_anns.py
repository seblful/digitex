import os

from digitex.core.anns_converter import OCRCaptionConverter

# Paths
HOME = os.getcwd()

TRAINING_DIR = os.path.join(HOME, "src", "digitex", "training")

DB_REPVIT_DIR = os.path.join(TRAINING_DIR, "db-repvit")
OCR_DATA_JSON_PATH = os.path.join(DB_REPVIT_DIR, "data", "raw-data", "data.json")

SVTR_DIR = os.path.join(TRAINING_DIR, "svtr2")
RAW_DATA_DIR = os.path.join(SVTR_DIR, "data", "finetune", "raw-data")
CAPTION_DATA_JSON_PATH = os.path.join(RAW_DATA_DIR, "data.json")

# First set Source Storage local file to absolute local path LS_LOCAL_STORAGE_PATH
LS_MEDIA_DIR = "C:/Users/seblful/AppData/Local/label-studio/label-studio/media"
LS_LOCAL_STORAGE_PATH = os.path.join(LS_MEDIA_DIR, "converted", "word-recogniton")
OCR_IMAGES_DIR = os.path.join(LS_MEDIA_DIR, "upload", "20")


def main() -> None:
    anns_converter = OCRCaptionConverter(LS_LOCAL_STORAGE_PATH)
    anns_converter.convert(
        OCR_IMAGES_DIR, OCR_DATA_JSON_PATH, CAPTION_DATA_JSON_PATH, RAW_DATA_DIR
    )


if __name__ == "__main__":
    main()
