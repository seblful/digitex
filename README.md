# tg-testing

Testing system of students using telegram-bot.

## Project Structure

```
tg-testing/
├── extract/              # Data extraction utilities
├── modules/              # Core processing modules
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
