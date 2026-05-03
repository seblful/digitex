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
├── scripts/                  # DB schema and population scripts
├── tests/                    # Test suite
└── AGENTS.md                 # AI agent instructions
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
digitex-bot run

# Populate database from extraction output
uv run python scripts/populate_db.py
```

## Telegram Bot

The bot allows students to take centralized tests via Telegram:

1. **Start** — `/start` to register and select a subject
2. **Navigate** — Choose subject → year → option number
3. **Test** — Answer Part A (multiple choice 1-5) and Part B (text) questions
4. **Results** — Get instant score and mistake review

### Bot Setup

1. Get a bot token from [@BotFather](https://t.me/BotFather)
2. Add to `.env` (or `.env.development`):
   ```
   BOT_TOKEN=your_bot_token_here
   BOT_ADMIN_USER_ID=your_telegram_user_id
   ```
3. Populate the database:
   ```bash
   uv run python scripts/populate_db.py
   ```
4. Run the bot:
   ```bash
   digitex-bot run
   ```

## Configuration

This project uses Pydantic Settings with environment-specific `.env` files:

```
.env                  # Base/shared variables (gitignored)
.env.development      # Development overrides (gitignored)
.env.production       # Production overrides (gitignored)
.env.example          # Reference template (committed)
```

Set `ENVIRONMENT` or `APP_ENVIRONMENT` to switch:

```bash
# Development (default)
digitex-bot run

# Production
ENVIRONMENT=production digitex-bot run
```

See `.env.example` for all available variables and their defaults.

## Setup

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run extraction
uv run python -m extraction.run --help

# Run training
uv run python -m training.cli --help
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

See [AGENTS.md](AGENTS.md) for:

- Code standards and conventions
- Type hinting requirements
- Testing guidelines
- Git workflow

## Testing

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_handlers.py
```
