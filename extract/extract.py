import os
import yaml

from components.extractor import ExtractorApp

INPUTS_DIR = "inputs"
CONFIG_PATH = os.path.join(INPUTS_DIR, "config.yaml")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config(CONFIG_PATH)
    app = ExtractorApp(cfg=config)
    app.run()


if __name__ == "__main__":
    main()
