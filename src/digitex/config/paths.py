"""Where the non-code inputs and outputs live."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsSettings(BaseSettings):
    """Where the non-code inputs and outputs live.

    Everything the project reads or writes that is not source — the book
    archive, the extraction output tree, model weights, training data — sits
    under ``data_root``. Code never derives a data path from its own location,
    so the package behaves the same installed into a container as it does run
    out of a checkout.

    ``data_root`` is resolved against the working directory, which is
    deliberate: an installed package cannot find the checkout it was built
    from, and guessing is exactly what the old ``BASE_DIR`` did. Set
    ``PATH_DATA_ROOT`` to be explicit. Every command that needs a directory
    reports the resolved path when it is missing, so the wrong cwd fails loudly
    instead of quietly extracting nothing.
    """

    model_config = SettingsConfigDict(env_prefix="PATH_", extra="ignore")

    data_root: Path = Field(
        default=Path("var"),
        description="Root of every non-code input and output.",
    )

    # Where the bot resolves an image's object_key against. Production rsyncs
    # the corpus to a directory of its own and bind-mounts it, so this is not
    # derivable from data_root there; unset, it is the extraction output tree
    # the keys were written from, which is what a laptop wants.
    questions_dir: Path | None = None

    # Repo content, not data: these YAMLs are hand-tuned hyperparameters under
    # version control, so they stay in the checkout rather than the data root.
    training_configs_dir: Path = Field(
        default=Path("configs/training"),
        description="Holds the {name}_train.yaml / {name}_val.yaml pair.",
    )

    @field_validator("data_root", "questions_dir", "training_configs_dir")
    @classmethod
    def _absolute(cls, value: Path | None) -> Path | None:
        """Pin relative paths to the working directory once, at load time.

        Resolving here rather than at each use means an error names a real
        absolute path, and nothing later depends on the cwd staying put.
        """
        return value.resolve() if value is not None else None

    # Top-level directories

    @computed_field
    @cached_property
    def books_dir(self) -> Path:
        return self.data_root / "books"

    @computed_field
    @cached_property
    def models_dir(self) -> Path:
        return self.data_root / "models"

    @computed_field
    @cached_property
    def extraction_dir(self) -> Path:
        return self.data_root / "extraction"

    # Extraction sub-paths

    @computed_field
    @cached_property
    def extraction_output_dir(self) -> Path:
        return self.extraction_dir / "output"

    @computed_field
    @cached_property
    def question_images_dir(self) -> Path:
        """Root that a question image's stored ``object_key`` resolves against."""
        return self.questions_dir or self.extraction_output_dir

    @computed_field
    @cached_property
    def extraction_model_path(self) -> Path:
        return self.models_dir / "page.pt"

    # Training sub-paths

    @computed_field
    @cached_property
    def training_data_dir(self) -> Path:
        return self.data_root / "training" / "data"
