# Digitex

Document digitization toolkit for processing images, ML-based document segmentation, and centralized testing via Telegram bot.

## Project Structure

The package is layered, and the layering is enforced rather than described —
see [`[tool.importlinter]`](pyproject.toml) and `tests/contracts/`. Only the bot
deploys, so nothing it can reach may touch a dependency the production image
does not install.

```
digitex/
├── src/digitex/
│   ├── domain/               # pure: entities, answer rules, corpus layout, geometry
│   ├── db/                   # Postgres: pool, unit of work, repositories, seed
│   │   └── migrations/       # Alembic scripts, shipped inside the package
│   ├── bot/                  # Telegram bot (aiogram)        ← the deploy layer
│   ├── imaging/              # OpenCV / Pillow / Tesseract
│   ├── ml/                   # YOLO segmentation and training
│   ├── labeling/             # Label Studio client
│   ├── pipeline/             # books → question images (+ audit/)
│   ├── ui/                   # the Tk review window
│   ├── config/               # settings, one module per layer
│   └── cli/                  # entry points
├── configs/training/         # YOLO hyperparameter YAMLs
├── deploy/                   # Dockerfile, deploy.sh, seed_prod.ps1
├── docker-compose.yml        # stays at the root: its relative paths are $APP_DIR
├── tools/                    # one-off local scripts
├── tests/                    # unit, integration, contracts
├── docs/                     # setup, deployment, workflows (see docs/README.md)
├── var/                      # the data root — gitignored, PATH_DATA_ROOT
└── CLAUDE.md                 # AI agent instructions
```

Dependencies point one way, and the two branches never meet:

```
cli ──▶ ui ──▶ pipeline ──▶ labeling ──▶ ml ──▶ imaging ──┐
 │                                                        ├──▶ domain
 └────▶ bot ──▶ db ─────────────────────────────────────  ┘
```

### The data root

Nothing outside `src/` is code. The book archive, extraction output, model
weights and training data all live under one directory — `var/` by default,
or wherever `PATH_DATA_ROOT` points — so the checkout stays small and an
installed package never guesses where a corpus is.

```
var/books/{subject}/raw/pages/{year}/…       scanned pages, the raw input
var/books/{subject}/processed/pages/{year}/… the same pages, corrected
var/books/{subject}/topics.json              the topic map, seeded into the db
var/extraction/output/…               question images, the corpus the bot serves
var/models/page.pt                    the segmentation checkpoint
var/training/{data,runs}/             YOLO datasets and run outputs
```

## Features

- **Image Extraction**: Extract and process question images from book images
- **Image Processing**: Crop, transform, resize with aspect ratio preservation
- **YOLO Segmentation**: Detect and segment document regions
- **Telegram Bot**: Take centralized tests via Telegram with automatic grading
- **Configuration Management**: Pydantic-based settings with environment variable support

## CLI Commands

```bash
# Prepare raw scans into var/books/{subject}/processed/ (renames, then corrects)
digitex-extract preprocess-scans

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

# Check the image rows still match the files on disk
uv run digitex-db check-images
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
# Everything (--no-extra cu130 on a machine without an NVIDIA GPU)
uv sync --all-extras --no-extra cpu

# Or just the workflow you are on
uv sync --extra pipeline      # books → question images
uv sync --extra ml            # YOLO training and prediction
uv sync --extra labeling      # the annotation server and its SDK

# Run extraction
uv run digitex-extract --help

# Run training
uv run digitex-train --help
```

The bot needs no extra at all: its dependencies are the base
`[project.dependencies]`, which is exactly what the production image installs.

### Requirements

- Python 3.13+
- uv package manager

## Modules

- **domain**: exam entities, answer matching, corpus layout, polygon spaces —
  pure Python and pydantic, importable by anything
- **db**: the only layer that writes SQL; one repository per aggregate, all
  writes through a `UnitOfWork`
- **bot**: the aiogram bot — the only thing that deploys
- **pipeline**: question images out of book scans, page → book → subject, plus
  `audit/` to check what came out
- **ml**: YOLO segmentation, dataset building and training
- **labeling**: Label Studio client and prediction upload
- **imaging**: cropping, deskewing, resizing, OCR
- **ui**: the Tk page-review window, the only package that imports tkinter

## Development

See [CLAUDE.md](CLAUDE.md) for code standards, type-hinting requirements,
testing guidelines, and git workflow.

## Testing

```bash
# Everything: unit, integration (needs Docker for Postgres), contracts
uv run pytest

# The unit suite alone — no Docker required
uv run pytest tests/unit

# One file
uv run pytest tests/unit/test_bot_handlers.py

# The deploy boundary, statically
uv run lint-imports
```

`tests/contracts/` is only meaningful in an environment built the way
production is, which is what the `deploy boundary` CI job does:

```bash
UV_PROJECT_ENVIRONMENT=.venv-prod uv sync --locked --no-dev --group contracts
UV_PROJECT_ENVIRONMENT=.venv-prod uv run --no-sync pytest tests/contracts -o addopts=""
```
