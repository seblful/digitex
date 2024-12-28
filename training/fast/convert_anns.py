import os

from modules.anns_converter import OCRBBOXAnnsConverter, OCRCaptionConverter

# Paths
HOME = os.getcwd()
DATA_DIR = os.path.join(HOME, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw-data")
DATA_JSON_PATH = os.path.join(RAW_DIR, "data.json")

LS_UPLOAD_DIR = "C:/Users/seblful/AppData/Local/label-studio/label-studio/media/upload"


def main():
    # # Convert ocr to bbox
    # ocr_bbox_anns_converter = OCRBBOXAnnsConverter(ls_upload_dir=LS_UPLOAD_DIR)
    # ocr_bbox_anns_converter.convert(input_json_path=DATA_JSON_PATH,
    #                                 output_dir=RAW_DIR)

    # Convert ocr to caption
    ocr_caption_converter = OCRCaptionConverter(ls_upload_dir=LS_UPLOAD_DIR)
    ocr_caption_converter.convert(input_json_path=DATA_JSON_PATH,
                                  output_project_num=11,
                                  output_dir=RAW_DIR)


if __name__ == "__main__":
    main()
