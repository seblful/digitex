# tg-testing

Testing system of students using telegram-bot.

## Project Structure

```
tg-testing/
├── extract/              # Data extraction utilities
├── modules/              # Core processing modules
│   ├── config/           # Application settings and configuration
│   │   └── settings.py   # Pydantic settings classes
│   ├── anns_converter.py
│   ├── data_creator.py
│   ├── handlers.py
│   ├── processors.py
│   ├── utils.py
│   └── predictors/       # ML predictors
│       ├── abstract_predictor.py
│       ├── prediction_result.py
│       └── segmentation.py  # YOLO segmentation predictor
├── training/            # Model training scripts
│   └── yolo/            # YOLO-based training
│       ├── components/  # Training utilities
│       └── data/        # Training data
└── AGENTS.md            # AI agent instructions
```

## Features

- **PDF Processing**: Extract and process PDF documents
- **Image Processing**: Crop, transform, and prepare images for ML
- **YOLO Segmentation**: Detect and segment document regions (pages)
- **Data Creation**: Generate training datasets from raw PDFs
- **Configuration Management**: Pydantic-based settings with environment variable support

## Configuration

This project uses Pydantic Settings for configuration management. Settings can be customized through:

1. **Environment Variables**: Create a `.env` file (see `.env.example` for reference)
2. **Direct Modification**: Defaults in `modules/config/settings.py`

### Settings Categories

- **AppSettings**: Application-wide constants (render scales, crop offsets, log level)
- **DatabaseSettings**: Database connection settings
- **TrainingSettings**: YOLO training parameters (epochs, batch size, etc.)
- **PathsSettings**: Directory paths for data, models, and datasets

### Example Usage

```python
from modules.config import get_settings

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
- `TRAIN_BATCH_SIZE`: Training batch size (default: 16)
- `TRAIN_IMAGE_SIZE`: Training image size (default: 640)

For a complete list, see `.env.example`.

## Setup

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run scripts
uv run <script_path>
```

### Requirements

- Python 3.13+
- uv package manager

## Modules

### DataCreator

Core class for creating training data from PDFs and images. Supports:

- Extracting pages from PDFs
- Extracting questions from annotated pages
- Extracting parts (text, tables, etc.) from questions
- Extracting words from parts

### Predictors

- **YOLO_SegmentationPredictor**: Segmentation using YOLO models

## Training

YOLO-based training pipeline for document segmentation:

```bash
cd training/yolo
uv run create_train_data.py  # Create training data
uv run train.py              # Train model
```

## Development

See [AGENTS.md](AGENTS.md) for:

- Code standards and conventions
- Type hinting requirements
- Testing guidelines
- Git workflow
