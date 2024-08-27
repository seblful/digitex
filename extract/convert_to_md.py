import os

import pypdfium2

from marker.models import load_all_models
from marker.convert import convert_single_pdf
from marker.output import save_markdown


# For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]


# Paths
HOME = os.getcwd()
DATA_DIR = os.path.join(HOME, "data")
INPUT_DIR = os.path.join(DATA_DIR, "inputs")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")

LANGS = ["ru", "en"]


def main() -> None:
    # Load models
    model_lst = load_all_models()

    for pdf_name in os.listdir(INPUT_DIR):
        # Check if it was converted before
        if not os.path.exists(os.path.join(OUTPUT_DIR, os.path.splitext(pdf_name)[0])):
            full_pdf_path = os.path.join(INPUT_DIR, pdf_name)

            # Convert pdf
            full_text, images, out_meta = convert_single_pdf(
                full_pdf_path, model_lst, langs=LANGS,  ocr_all_pages=True)

            # Save markdown
            subfolder_dir = save_markdown(
                OUTPUT_DIR, pdf_name, full_text, images, out_meta)
            print(f"Saved markdown to the '{
                os.path.basename(subfolder_dir)}' folder.")

            break


if __name__ == "__main__":
    main()
