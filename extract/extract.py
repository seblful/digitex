import os
import yaml

from components.extractor import ExtractorApp

CONFIG_DIRECTORY = "inputs"
CONFIG_FILE = os.path.join(CONFIG_DIRECTORY, "config.yaml")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config(CONFIG_FILE)
    app = ExtractorApp(cfg=config)
    app.run()


if __name__ == "__main__":
    main()
