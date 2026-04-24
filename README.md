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
2. Add to `.env`:
   ```
   BOT__TOKEN=your_bot_token_here
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

This project uses Pydantic Settings for configuration management. Settings can be customized through:

1. **Environment Variables**: Create a `.env` file (see `.env.example` for reference)
2. **Direct Modification**: Defaults in `src/digitex/config/settings.py`

### Settings Categories

- **ExtractionSettings**: Image extraction parameters (model path, output format, image dimensions)
- **DatabaseSettings**: Database connection settings
- **TrainingSettings**: YOLO training parameters (epochs, batch size, image size, etc.)
- **PathsSettings**: Directory paths for data, models, and datasets
- **BotSettings**: Telegram bot token

### Environment Variables

Key environment variables include:

- `BOT__TOKEN`: Telegram bot token from @BotFather
- `EXTRACTION_MODEL_PATH`: Path to the YOLO segmentation model (default: extraction/models/page.pt)
- `EXTRACTION_BOOKS_DIR`: Source books directory (default: books)
- `EXTRACTION_EXTRACTION_DIR`: Output directory (default: extraction/output)
- `EXTRACTION_QUESTION_MAX_WIDTH`: Maximum width for extracted questions (default: 2000)
- `EXTRACTION_QUESTION_MAX_HEIGHT`: Maximum height for extracted questions (default: 2000)
- `EXTRACTION_IMAGE_FORMAT`: Output image format (default: jpg)
- `TRAIN_NUM_EPOCHS`: Training epochs (default: 100)
- `TRAIN_BATCH_SIZE`: Training batch size (default: 4)
- `TRAIN_IMAGE_SIZE`: Training image size (default: 640)
- `DB_PATH`: Database file path (default: data/tests.db)

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
