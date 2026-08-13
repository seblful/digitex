"""Every module the deployed bot can reach must import in the production image.

Only the Telegram bot ships. Its image installs ``[project.dependencies]`` and
nothing else — no OpenCV, no torch, no Tesseract — so a single stray import in
bot or database code is an ImportError on the VPS. Nothing else catches it: the
normal CI job installs every extra, the Docker build imports no application
code, and the image's own smoke test only runs ``--help``.

So this suite exists to be run **in a production-shaped environment**, where a
missing dependency is what makes it fail:

    uv sync --locked --no-dev --group contracts
    uv run --no-sync pytest tests/contracts

That is the whole assertion, and it is deliberately not clever. The static
half of the rule — which package may import which, transitively — is
``[tool.importlinter]`` in pyproject.toml, which says it better and runs on
every commit. This is the half that proves the declared production dependency
list is actually sufficient to run the bot, rather than a list someone kept up
to date by hand.

Walking every module matters because the entry points import lazily inside
functions — ``cli/bot.py`` imports aiogram inside ``_main``, ``cli/db.py``
imports the seed loader inside its command — so ``import digitex.cli.bot`` alone
would sail straight past the imports most likely to be wrong.

In the development environment every extra is installed, so this passes
trivially. That is expected: the environment is the test.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

# The packages the production image installs and the bot process runs. Anything
# absent from this list is a local-workflow package, and is not in the image.
# Keep in sync with the source_modules of both forbidden [tool.importlinter]
# contracts in pyproject.toml.
DEPLOY_PACKAGES = (
    "digitex.bot",
    "digitex.cli._shared",
    "digitex.cli.bot",
    "digitex.cli.db",
    "digitex.config",
    "digitex.db",
    "digitex.domain",
    "digitex.logging",
)


def _submodules(package_name: str) -> list[str]:
    """Every importable module at or under *package_name*."""
    package = importlib.import_module(package_name)
    spec = getattr(package, "__spec__", None)
    if spec is None or not spec.submodule_search_locations:
        return [package_name]  # a plain module, not a package

    found = [package_name]
    for info in pkgutil.walk_packages(
        spec.submodule_search_locations, prefix=f"{package_name}."
    ):
        # digitex.db.migrations holds Alembic scripts, which Alembic executes
        # against a live connection rather than importing.
        if ".migrations" in info.name:
            continue
        found.append(info.name)
    return found


def _deploy_modules() -> list[str]:
    modules: list[str] = []
    for package in DEPLOY_PACKAGES:
        modules.extend(_submodules(package))
    return sorted(set(modules))


DEPLOY_MODULES = _deploy_modules()


def test_the_walk_found_the_bot() -> None:
    """Guard the guard: a typo in DEPLOY_PACKAGES would empty the suite."""
    assert "digitex.bot.renderer" in DEPLOY_MODULES
    assert "digitex.db.repositories.question" in DEPLOY_MODULES
    assert len(DEPLOY_MODULES) > 30


@pytest.mark.parametrize("module_name", DEPLOY_MODULES)
def test_module_imports_with_production_dependencies_only(module_name: str) -> None:
    """Fails with ImportError when run without the extras installed."""
    assert importlib.import_module(module_name) is not None
