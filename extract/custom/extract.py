import os
import logging

from extractor import TestExtractor

# Paths
HOME = os.getcwd()

TESTING_DIR = os.path.dirname(os.path.dirname(HOME))
PDF_DIR = os.path.join(TESTING_DIR, "raw-data/biology/new")
INPUTS_DIR = os.path.join(HOME, "inputs")
OUTPUTS_DIR = os.path.join(HOME, "outputs")

LOG_LEVEL = logging.DEBUG


def main() -> None:
    extractor = TestExtractor(pdf_dir=PDF_DIR,
                              inputs_dir=INPUTS_DIR,
                              outputs_dir=OUTPUTS_DIR,
                              langs=["ru", "en"],
                              log_level=LOG_LEVEL)
    extractor.extract()


if __name__ == "__main__":
    main()
