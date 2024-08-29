import os

from processors import PDFProcessor

# For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]

# Paths
HOME = os.getcwd()

TESTING_DIR = os.path.dirname(os.path.dirname(HOME))
RAW_DIR = os.path.join(TESTING_DIR, "raw-data")

DATA_DIR = os.path.join(HOME, "data")
INPUT_DIR = os.path.join(DATA_DIR, "inputs")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")


def main() -> None:
    pdf_processor = PDFProcessor(raw_dir=RAW_DIR,
                                 input_dir=INPUT_DIR,
                                 output_dir=OUTPUT_DIR,
                                 alpha=2, beta=10,
                                 remove_ink=True,
                                 binarize=False,
                                 blur=False)
    pdf_processor.process()


if __name__ == "__main__":
    main()
