FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN uv pip install --system \
    aiogram>=3.27.0 \
    structlog>=25.5.0 \
    pydantic>=2.0 \
    pydantic-settings>=2.0 \
    typer

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "digitex.cli.bot"]
