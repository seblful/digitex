import os
import logging


class TestExtractor:
    def __init__(self,
                 raw_data_dir: str,
                 models_dir: str,
                 outputs_dir: str,
                 langs: list = ["ru", "en"]) -> None:
        # Paths
        self.raw_data_dir = raw_data_dir
        self.models_dir = models_dir
        self.outputs_dir = outputs_dir

        # Language
        self.langs = langs

        # Setup logging
        self.__setup_logging()

    def __setup_logging(self):
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s %(levelname)s %(message)s",
                            datefmt="%d/%m/%Y %H:%M:%S",
                            filename="basic.log")
        logging.info("Logging is configured.")

    def extract(self) -> None:
        pass
