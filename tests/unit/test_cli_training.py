"""Tests for the training CLI's argument contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from digitex.cli import training

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture(autouse=True)
def _quiet_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the Typer callback from configuring real logging."""
    monkeypatch.setattr(training, "setup_logging", lambda _settings: None)


class TestCreateDataset:
    @pytest.mark.parametrize("split", ["-0.5", "1.5"], ids=["negative", "above-one"])
    def test_train_split_outside_the_unit_interval_is_refused(self, split: str) -> None:
        """An out-of-range split must die at the option, not in the math.

        A split like -0.5 used to reach the partition arithmetic and produce
        train/test sets that overlap — the model then validates on data it
        trained on, with no warning.
        """
        result = runner.invoke(
            training.app, ["create-dataset", "page", "--train-split", split]
        )

        assert result.exit_code != 0
        assert "0.0" in result.output
        assert "1.0" in result.output
