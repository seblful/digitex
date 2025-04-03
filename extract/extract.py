import os
import yaml

from components.extractor import ExtractorApp

INPUTS_DIR = "inputs"
CONFIG_PATH = os.path.join(INPUTS_DIR, "config.yaml")


def main() -> None:
    with open(CONFIG_PATH) as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    app = ExtractorApp(cfg=cfg)
    app.run()


if __name__ == "__main__":
    main()
