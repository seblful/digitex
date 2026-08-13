"""Fixtures shared by the unit suite.

What every unit test needs is an environment that says nothing, so the
assertions are about the code's own defaults rather than about the machine
running them.

``Settings.load()`` calls ``load_dotenv``, which copies this machine's ``.env``
into ``os.environ`` and leaves it there for the rest of the process. Without
this fixture the first test to resolve settings decides what every test after it
reads — which is how ``test_default_dsn`` came to pass on CI, pass on its own,
and fail in the full suite on a developer's laptop.

The integration suite deliberately points ``DATABASE_URL`` at its container, so
this belongs here rather than at the root.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from digitex.config import reset_settings_cache

if TYPE_CHECKING:
    from collections.abc import Iterator

# Every prefix a settings group claims, plus the bare names listed as aliases
# on a field. Keep in sync with the ``env_prefix`` of each group in
# ``digitex.config`` and the ``AliasChoices`` on DatabaseSettings.dsn and
# AppSettings.environment.
_ENV_PREFIXES = (
    "APP_",
    "BOT_",
    "DATA_",
    "DB_",
    "EXTRACTION_",
    "LABEL_STUDIO_",
    "LOGGING_",
    "OPENROUTER_",
    "PATH_",
    "TIMEZONE_",
)
_ENV_NAMES = ("DATABASE_URL", "ENVIRONMENT")


@pytest.fixture(autouse=True)
def _settings_defaults(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start each test from the settings defaults, whatever the machine sets.

    Clearing at setup is what does the work: a test that resolves settings
    re-populates ``os.environ`` behind monkeypatch's back, so the next test
    cleans up after it rather than trusting it to.
    """
    for name in list(os.environ):
        if name.startswith(_ENV_PREFIXES) or name in _ENV_NAMES:
            monkeypatch.delenv(name, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()
