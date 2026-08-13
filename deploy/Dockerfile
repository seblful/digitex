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

# Dependencies alone first, so editing src/ does not re-resolve or re-download
# anything. `--no-dev` with no extras is the production set from pyproject.toml
# — locked, so this image cannot drift from the commit that built it.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

# README.md is here because pyproject.toml names it as the project readme, and
# the build backend reads it. alembic.ini and migrations/ need no COPY of their
# own — they live inside src/digitex/db/ and install with the package.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Installs the project itself, which puts digitex-bot and digitex-db on PATH.
# --no-editable, so the venv holds a built copy of the package and nothing
# resolves a path by walking up from a source file. That leaves /app/src
# unreferenced at runtime; the final stage does not carry it.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM ${PYTHON_IMAGE}

# A fixed uid, because the deploy chowns the mounted logs directory to it.
RUN groupadd --system --gid 10001 digitex \
    && useradd --system --uid 10001 --gid digitex --home-dir /app digitex

WORKDIR /app

# The virtualenv and nothing else. Because the builder installed the project
# --no-editable, the venv holds a real copy of the package — including
# alembic.ini and migrations/, which digitex.db.schema resolves through
# importlib.resources — so /app/src, pyproject.toml and uv.lock have no runtime
# reader and stay out of the image.
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER digitex

CMD ["digitex-bot"]
