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
# the build backend reads it.
COPY pyproject.toml uv.lock alembic.ini README.md ./
COPY src/ ./src/
COPY migrations/ ./migrations/

# Installs the project itself, which puts digitex-bot and digitex-db on PATH.
# The install stays editable on purpose: config.BASE_DIR walks up from
# digitex/config.py to find alembic.ini and migrations/, and only the source
# layout puts them where it looks.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

FROM ${PYTHON_IMAGE}

# A fixed uid, because the deploy chowns the mounted logs directory to it.
RUN groupadd --system --gid 10001 digitex \
    && useradd --system --uid 10001 --gid digitex --home-dir /app digitex

WORKDIR /app
COPY --from=builder --chown=10001:10001 /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER digitex

CMD ["digitex-bot"]
