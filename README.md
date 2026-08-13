# Digitex

Document digitization toolkit for processing images, ML-based document segmentation, and centralized testing via Telegram bot.

## Project Structure

```
digitex/
├── src/digitex/              # Main package
│   ├── bot/                  # Telegram bot (aiogram)
│   ├── config/               # Configuration management
│   ├── core/                 # Core domain logic + DB layer
│   ├── ml/                   # Machine learning components
│   ├── extractors/           # Image extraction pipeline
│   └── cli/                  # CLI entry points
├── extraction/               # Legacy extraction scripts
├── training/                 # ML training workflow
├── migrations/               # Alembic schema migrations
├── scripts/                  # DB population scripts
├── tests/                    # Test suite
├── docs/                     # Setup, deployment, workflows (see docs/README.md)
└── CLAUDE.md                 # AI agent instructions
```

## Features

- **Image Extraction**: Extract and process question images from book images
- **Image Processing**: Crop, transform, resize with aspect ratio preservation
- **YOLO Segmentation**: Detect and segment document regions
- **Telegram Bot**: Take centralized tests via Telegram with automatic grading
- **Configuration Management**: Pydantic-based settings with environment variable support

## CLI Commands

```bash
# Extract question images from books
digitex-extract extract-questions <subject>

# Train YOLO segmentation model
digitex-train create-dataset
digitex-train train

# Start Telegram bot
digitex-bot

# Manage schema migrations
uv run digitex-db upgrade

# Populate database from extraction output
uv run digitex-db populate
```

## Telegram Bot

The bot allows students to take centralized tests via Telegram:

1. **Start** — `/start` to register and select a subject
1. **Navigate** — Choose subject → year → option number
1. **Test** — Answer Part A (multiple choice 1-5) and Part B (text) questions
1. **Results** — Get instant score and mistake review

### Setup & deployment

- **Run locally**: see [docs/local-setup.md](docs/local-setup.md)
- **Deploy to a VPS / day-2 ops**: see [docs/production.md](docs/production.md)
- **Migration CLI & schema conventions**: see [docs/database-reference.md](docs/database-reference.md)

Full doc index: [docs/README.md](docs/README.md).

## Configuration

This project uses Pydantic Settings with one `.env` file per machine:

```
.env                  # This machine's config (gitignored)
.env.example          # Reference template (committed)
```

Your laptop's `.env` holds development values, the server's holds production
ones — there is no second file to keep in sync. Real environment variables win
over the file, which is how Compose injects `DATABASE_URL` and sets
`ENVIRONMENT=production` to select the JSON log renderer.

See `.env.example` for all available variables and their defaults.

## Setup

This project uses `uv` for dependency management.

```bash
# Install dependencies (--no-extra cu130 on a machine without an NVIDIA GPU)
uv sync --all-extras --no-extra cpu

# Run extraction
uv run digitex-extract --help

# Run training
uv run digitex-train --help
```

### Requirements

- Python 3.13+
- uv package manager

## Modules

- **extractors**: Extract question images from book images using YOLO segmentation
- **creators**: Create training data from raw images
- **core**: Handlers, processors, and core utilities
- **ml**: YOLO-based segmentation models and training

## Development

See [CLAUDE.md](CLAUDE.md) for code standards, type-hinting requirements,
testing guidelines, and git workflow.

## Testing

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_handlers.py
```
