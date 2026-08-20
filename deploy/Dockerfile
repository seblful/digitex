# syntax=docker/dockerfile:1

# Pinned by digest so an image is reproducible from its commit; .github/
# dependabot.yml opens the bump PRs.
ARG PYTHON_IMAGE=python:3.13-slim@sha256:ffb752e139c0a19692a43af8d8523b274222dd68eebad5d583b45c2201c6e30a

FROM ${PYTHON_IMAGE} AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies alone first, so editing a package does not re-resolve or
# re-download anything.
#
# `uv sync --no-dev` at the workspace root installs the root's dependencies —
# `digitex-core` and `digitex-service` — and nothing else. `digitex-studio`
# hangs off the `studio` extra, so OpenCV, torch and Tesseract are not merely
# forbidden in this image, they are never resolved. That is the deploy boundary:
# a fact about this dependency graph rather than a rule someone maintains.
#
# Every member's pyproject.toml has to be present for the lock to resolve, even
# the one that is not installed.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/core/pyproject.toml,target=packages/core/pyproject.toml \
    --mount=type=bind,source=packages/service/pyproject.toml,target=packages/service/pyproject.toml \
    --mount=type=bind,source=packages/studio/pyproject.toml,target=packages/studio/pyproject.toml \
    uv sync --locked --no-dev --no-install-workspace

# README.md is here because the workspace root names it as the project readme.
# alembic.ini and migrations/ need no COPY of their own — they live inside
# packages/service/src/digitex/db/ and install with the package.
COPY pyproject.toml uv.lock README.md ./
COPY packages/ ./packages/

# Installs the two shipped members, which puts digitex-bot and digitex-db on
# PATH. --no-editable, so the venv holds a built copy of each package and
# nothing resolves a path by walking up from a source file. That leaves
# /app/packages unreferenced at runtime; the final stage does not carry it.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM ${PYTHON_IMAGE}

# A fixed uid, because the deploy chowns the mounted logs directory to it.
RUN groupadd --system --gid 10001 digitex \
    && useradd --system --uid 10001 --gid digitex --home-dir /app digitex

WORKDIR /app

# The virtualenv and nothing else. Because the builder installed the members
# --no-editable, the venv holds a real copy of each — including alembic.ini and
# migrations/, which digitex.db.schema resolves through importlib.resources — so
# /app/packages, pyproject.toml and uv.lock have no runtime reader and stay out
# of the image.
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER digitex

CMD ["digitex-bot"]
