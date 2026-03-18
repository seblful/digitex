# Training Guide

This guide covers how to create datasets and train YOLO segmentation models for document analysis.

## Prerequisites

- Python 3.13
- [UV](https://github.com/astral-sh/uv) package manager

## Workflow

### 1. Create PDFs from Images

If you have source images and need to convert them to PDFs first:

```bash
uv run python -c "
from pathlib import Path
from digitex.utils import create_pdf_from_images

# Convert images in a directory to PDF
create_pdf_from_images(
    image_dir=Path('path/to/images'),
    raw_dir=Path('path/to/output'),
    process=False
)
"
```

### 2. Extract Pages from PDFs

Extract random pages from PDFs to create training images:

```bash
uv run python -c "
from pathlib import Path
from digitex.core import PageDataCreator

creator = PageDataCreator()
creator.create(
    pdf_dir=Path('path/to/pdfs'),
    output_dir=Path('path/to/output'),
    num_images=100
)
"
```

### 3. Annotate Images

Use [Label Studio](https://labelstud.io/) to annotate the extracted images with polygon segmentation labels.

Organize your data as follows:

```
data/
├── page/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── labels/
│   │   ├── image_001.txt
│   │   ├── image_002.txt
│   │   └── ...
│   └── classes.txt
```

Format for `classes.txt`:

```
question
answer
number
option
spec
```

Format for label files (YOLO polygon format):

```
<class_idx> <x1> <y1> <x2> <y2> <x3> <y3> ...
```

### 4. Create Dataset

Convert annotated images to YOLO dataset format:

```bash
uv run python cli/create_dataset.py \
    --data_subdir page \
    --train_split 0.8 \
    --anns_type polygon
```

Options:

- `--data_subdir`: Task type (e.g., `page`, `question`, `part`)
- `--train_split`: Train/val split ratio (default: 0.8)
- `--anns_type`: Annotation type (default: `polygon`)
- `--augment`: Enable augmentation
- `--aug_images`: Number of augmented images (default: 100)
- `--visualize`: Enable visualization
- `--vis_images`: Number of images to visualize (default: 50)

### 5. Train Model

Train a YOLO segmentation model:

```bash
uv run python cli/train.py
```

Or with custom parameters:

```bash
uv run python cli/train.py \
    --data_subdir page \
    --model_type seg \
    --model_size m \
    --num_epochs 100 \
    --image_size 640 \
    --batch_size 4
```

Training parameters (all can be overridden via CLI or environment variables):

| Parameter             | CLI Flag                  | Environment                   | Default | Description                          |
| --------------------- | ------------------------- | ----------------------------- | ------- | ------------------------------------ |
| data_subdir           | `--data-subdir`           | `TRAIN_DATA_SUBDIR`           | `page`  | Task type                            |
| model_type            | `--model-type`            | `TRAIN_MODEL_TYPE`            | `seg`   | YOLO model type                      |
| model_size            | `--model-size`            | `TRAIN_MODEL_SIZE`            | `m`     | Model size (`n`, `s`, `m`, `l`, `x`) |
| num_epochs            | `--num-epochs`            | `TRAIN_NUM_EPOCHS`            | `100`   | Number of epochs                     |
| image_size            | `--image-size`            | `TRAIN_IMAGE_SIZE`            | `640`   | Input image size                     |
| batch_size            | `--batch-size`            | `TRAIN_BATCH_SIZE`            | `4`     | Batch size                           |
| pretrained_model_path | `--pretrained-model-path` | `TRAIN_PRETRAINED_MODEL_PATH` | `None`  | Fine-tune from existing model        |
| overlap_mask          | `--overlap-mask`          | `TRAIN_OVERLAP_MASK`          | `False` | Use overlapping masks                |
| patience              | `--patience`              | `TRAIN_PATIENCE`              | `50`    | Early stopping patience              |
| seed                  | `--seed`                  | `TRAIN_SEED`                  | `42`    | Random seed                          |

## Model Sizes

| Size | Parameters | Speed   | Accuracy |
| ---- | ---------- | ------- | -------- |
| n    | 3.2M       | Fastest | Lowest   |
| s    | 11.2M      | Fast    | Lower    |
| m    | 25.9M      | Medium  | Medium   |
| l    | 47.5M      | Slow    | Higher   |
| x    | 99.8M      | Slowest | Highest  |

## Directory Structure

```
project/
├── books/                    # Source PDFs
├── training/                  # Training outputs (gitignored)
│   ├── data/
│   │   └── page/
│   │       ├── books/        # PDFs for training
│   │       ├── images/       # Extracted pages
│   │       ├── labels/      # YOLO format labels
│   │       ├── classes.txt   # Class names
│   │       ├── dataset/      # YOLO dataset format
│   │       │   ├── train/
│   │       │   ├── val/
│   │       │   └── data.yaml
│   │       └── check-images/ # Visualization output
│   ├── models/               # Trained models
│   │   └── page/
│   │       └── train/weights/best.pt
│   └── output/               # YOLO training runs
└── cli/
    ├── train.py
    └── create_dataset.py
```
