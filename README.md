# Digitex

Document digitization toolkit for processing images and ML-based document segmentation.

## Project Structure

```
digitex/
├── src/digitex/              # Main package
│   ├── config/               # Configuration management
│   ├── core/                # Core domain logic
│   ├── ml/                  # Machine learning components
│   └── utils.py            # Common utilities
├── extraction/              # Image extraction pipeline
├── training/               # ML training workflow
├── scripts/               # SQL scripts
├── tests/                 # Test suite
└── AGENTS.md             # AI agent instructions
```

## Features

- **Image Extraction**: Extract and process question images from book images
- **Image Processing**: Crop, transform, resize with aspect ratio preservation
- **YOLO Segmentation**: Detect and segment document regions
- **Configuration Management**: Pydantic-based settings with environment variable support

## Configuration

This project uses Pydantic Settings for configuration management. Settings can be customized through:

1. **Environment Variables**: Create a `.env` file (see `.env.example` for reference)
2. **Direct Modification**: Defaults in `src/digitex/config/settings.py`

### Settings Categories

- **ExtractionSettings**: Image extraction parameters (model path, output format, image dimensions)
- **DatabaseSettings**: Database connection settings
- **TrainingSettings**: YOLO training parameters (epochs, batch size, image size, etc.)
- **PathsSettings**: Directory paths for data, models, and datasets

### Example Usage

```python
from digitex.config import get_settings

settings = get_settings()

# Access extraction settings
image_format = settings.extraction.image_format
question_size = (settings.extraction.question_max_width, settings.extraction.question_max_height)

# Access training settings
image_size = settings.training.image_size
num_epochs = settings.training.num_epochs

# Access database settings
db_path = settings.database.path
```

### Environment Variables

Key environment variables include:

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
