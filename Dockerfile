FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/script.sql ./scripts/
COPY data/development.db /app/seed/seed.db

RUN uv pip install --system \
    aiogram>=3.27.0 \
    structlog>=25.5.0 \
    pydantic>=2.0 \
    pydantic-settings>=2.0 \
    typer

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

RUN mkdir -p data logs /app/seed

VOLUME ["/app/data"]

COPY scripts/docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
