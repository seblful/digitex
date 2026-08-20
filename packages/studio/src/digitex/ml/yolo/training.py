"""Training a YOLO segmentation model from its config pair.

``run`` owns the whole recipe: read the base model out of the train config,
train, then validate. The order is not the caller's to choose — ``val`` reads
the weights ``train`` writes — which is why the two are one call rather than
two the caller has to sequence.

Everything else about a run lives in the YAML that ultralytics reads. Nothing
here adds a default, overrides a setting, or looks at the results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()


def model_name(train_config: Path) -> str:
    """The base model a train config asks for.

    Raises:
        ValueError: If the config has no ``model`` key.
    """
    import yaml

    config = yaml.safe_load(train_config.read_text(encoding="utf-8")) or {}
    name = config.get("model")
    if not name:
        raise ValueError(f"'model' key missing in {train_config}")
    return str(name)


def run(train_config: Path, val_config: Path) -> None:
    """Train on *train_config*, then validate on *val_config*.

    Both configs are checked before anything loads, so a mistyped val config is
    caught now rather than once training has already finished.

    Raises:
        FileNotFoundError: If either config file is missing.
        ValueError: If the train config names no model.
    """
    for config in (train_config, val_config):
        if not config.exists():
            raise FileNotFoundError(config)

    name = model_name(train_config)

    # Imported here: ultralytics pulls in torch, which costs seconds even when
    # the caller only wanted --help.
    from ultralytics import YOLO  # type: ignore[import-untyped]

    logger.info("Starting YOLO training", model=name, train_config=str(train_config))
    model = YOLO(name)
    model.train(cfg=train_config)
    model.val(cfg=val_config)
    logger.info("Training and validation complete", model=name)
