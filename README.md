# Digitex

Document digitization toolkit for processing PDFs, images, and ML-based document segmentation.

## Project Structure

```
digitex/
├── src/digitex/              # Main package
│   ├── config/               # Configuration management
│   ├── core/                # Core domain logic
│   ├── ml/                  # Machine learning components
│   └── utils.py            # Common utilities
├── cli/                    # Command-line entry points
│   └── init_db.py         # Initialize database
├── training/               # ML training workflow
├── scripts/               # SQL scripts
├── tests/                 # Test suite
└── AGENTS.md             # AI agent instructions
```

## Features

- **PDF Processing**: Extract and process PDF documents
- **Image Processing**: Crop, transform, and prepare images
- **YOLO Segmentation**: Detect and segment document regions
- **Configuration Management**: Pydantic-based settings with environment variable support

## Configuration

This project uses Pydantic Settings for configuration management. Settings can be customized through:

1. **Environment Variables**: Create a `.env` file (see `.env.example` for reference)
2. **Direct Modification**: Defaults in `src/digitex/config/settings.py`

### Settings Categories

- **AppSettings**: Application-wide constants (render scales, crop offsets, log level)
- **DatabaseSettings**: Database connection settings
- **TrainingSettings**: YOLO training parameters (epochs, batch size, etc.)
- **PathsSettings**: Directory paths for data, models, and datasets

### Example Usage

```python
from digitex.config import get_settings

settings = get_settings()

# Access application settings
render_scale = settings.app.render_scale

# Access training settings
num_epochs = settings.training.num_epochs

# Access database settings
db_path = settings.database.path
```

### Environment Variables

Key environment variables include:

- `APP_RENDER_SCALE`: PDF rendering scale factor (default: 3)
- `APP_CROP_OFFSET`: Image crop border offset (default: 0.025)
- `APP_LOG_LEVEL`: Logging level (default: INFO)
- `DB_PATH`: Database file path (default: data/tests.db)
- `TRAIN_NUM_EPOCHS`: Training epochs (default: 100)
- `TRAIN_BATCH_SIZE`: Training batch size (default: 4)
- `TRAIN_IMAGE_SIZE`: Training image size (default: 640)

For a complete list, see `.env.example`.

## Setup

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run CLI scripts
uv run python -m training.cli --help
uv run python cli/init_db.py
```

### Requirements

- Python 3.13+
- uv package manager

## Modules

- **DataCreator**: Create training data from PDFs and images
- **Handlers**: PDF, image, and label file operations
- **Processors**: Image enhancement and file I/O
- **Predictors**: YOLO-based segmentation models

## Training

YOLO-based training pipeline for document segmentation. See [training/README.md](training/README.md) for detailed documentation.

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
