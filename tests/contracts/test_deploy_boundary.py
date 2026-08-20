"""Every module the deployed half can reach must import in the production image.

Only `digitex-core` and `digitex-service` ship. Their image installs those two
distributions and their dependencies — no OpenCV, no torch, no Tesseract — so a
single stray import in bot or database code is an ImportError on the VPS.
Nothing else catches it: the normal CI job installs the studio too, the Docker
build imports no application code, and the image's own smoke test only runs
``--help``.

So this suite exists to be run **in a production-shaped environment**, where a
missing dependency is what makes it fail:

    uv sync --locked --no-dev --group contracts
    uv run --no-sync pytest tests/contracts

Most of what this used to defend is now a packaging fact: `digitex-service`
does not depend on `digitex-studio`, so `digitex.pipeline` is not merely
forbidden in production, it is absent. What is left for a test is the half
packaging cannot state — that the declared dependency list is actually
*sufficient* to run the bot, rather than a list someone kept up to date by
hand.

Walking every module matters because the entry points import lazily inside
functions — `service/cli/bot.py` imports aiogram inside ``_main``,
`service/cli/db.py` imports the seed loader inside its command — so importing
the entry point alone would sail straight past the imports most likely to be
wrong.

In the development environment every package is installed, so this passes
trivially. That is expected: the environment is the test.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

# The top-level packages the two shipped distributions provide. The bot process
# runs out of these and nothing else.
#
# `test_the_list_covers_everything_the_shipped_members_hold` checks this against
# the workspace on disk, so it cannot silently fall behind a new package.
DEPLOY_PACKAGES = (
    "digitex.bot",
    "digitex.config",
    "digitex.console",
    "digitex.db",
    "digitex.domain",
    "digitex.logging",
    "digitex.service",
)

# Which workspace members ship. Used only by the coverage check below, which
# skips itself when the repo is not on disk (the production image has no
# `packages/` tree — it installs wheels).
SHIPPED_MEMBERS = ("core", "service")


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


def _workspace_root() -> Path | None:
    """The repo root, or None when running against installed wheels."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "packages").is_dir():
            return parent
    return None


def test_the_walk_found_the_bot() -> None:
    """Guard the guard: a typo in DEPLOY_PACKAGES would empty the suite."""
    assert "digitex.bot.renderer" in DEPLOY_MODULES
    assert "digitex.db.repositories.question" in DEPLOY_MODULES
    assert "digitex.service.cli.bot" in DEPLOY_MODULES
    assert len(DEPLOY_MODULES) > 30


def test_the_list_covers_everything_the_shipped_members_hold() -> None:
    """A new package in core or service has to be declared above.

    The failure this replaces: `DEPLOY_PACKAGES` used to carry a comment asking
    the next person to keep it in sync with two `[tool.importlinter]` contracts
    by hand. A package left out of it is not a test failure — it is silently
    less coverage — so the list is checked against what the members actually
    ship rather than trusted.
    """
    root = _workspace_root()
    if root is None:
        pytest.skip("installed as wheels; no workspace tree to compare against")

    on_disk = {
        f"digitex.{path.stem if path.is_file() else path.name}"
        for member in SHIPPED_MEMBERS
        for path in (root / "packages" / member / "src" / "digitex").iterdir()
        if path.suffix == ".py" or (path.is_dir() and path.name != "__pycache__")
    }
    assert on_disk == set(DEPLOY_PACKAGES)


@pytest.mark.parametrize("module_name", DEPLOY_MODULES)
def test_module_imports_with_production_dependencies_only(module_name: str) -> None:
    """Fails with ImportError when run without the studio installed."""
    assert importlib.import_module(module_name) is not None
