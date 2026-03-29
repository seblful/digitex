# Training

YOLO-based training pipeline for document segmentation.

**Note:** Run all commands from the project root directory.

## Directory Structure

```
training/
├── cli.py              # Unified CLI for training workflows
├── data/               # Training data (gitignored)
│   └── <task>/         # Task-specific data (page, question, part)
│       ├── raw-data/   # Raw annotations and images
│       ├── dataset/    # Processed YOLO dataset
│       └── images/     # Extracted page images
├── output/             # Training outputs (gitignored)
└── runs/               # Ultralytics runs (gitignored)
```

## Commands

### 1. Prepare Training Data

Select random images from book folders and save them for annotation:

```bash
uv run python -m training.cli prepare-train-data --data-subdir page --num-images 100
```

**Options:**
- `--data-subdir`: Task type (default: `page`)
- `--num-images`: Number of images to select (default: 100)

**Requirements:**
- Book images in `books/<subject>/images/<year>/` directory

### 2. Create Dataset

Convert raw annotations to YOLO dataset format:

```bash
uv run python -m training.cli create-data --data-subdir page --train-split 0.8
```

**Options:**
- `--data-subdir`: Task type (default: `page`)
- `--train-split`: Training/validation split ratio (default: 0.8)
- `--anns-type`: Annotation type (default: `polygon`)
- `--augment`: Enable data augmentation (default: False)
- `--aug-images`: Number of augmented images (default: 100)
- `--visualize`: Visualize dataset samples (default: False)
- `--vis-images`: Number of images to visualize (default: 50)

**Requirements:**
- Raw annotations in `training/data/<task>/raw-data/`

### 3. Train Model

Train YOLO segmentation model:

```bash
uv run python -m training.cli train --num-epochs 50
```

**Options:**
- `--data-subdir`: Task type (default from settings)
- `--model-type`: YOLO model type (default: `seg`)
- `--model-size`: Model size: n, s, m, l, x (default: `m`)
- `--num-epochs`: Training epochs (default: 100)
- `--image-size`: Input image size (default: 640)
- `--batch-size`: Batch size (default: 4)
- `--overlap-mask`: Mask overlap for segmentation (default: False)
- `--patience`: Early stopping patience (default: 50)
- `--seed`: Random seed for reproducibility (default: 42)
- `--pretrained-model-path`: Path to existing model weights

## Complete Workflow

Run from the project root directory:

```bash
# Step 1: Prepare training data (select random images from books)
uv run python -m training.cli prepare-train-data --num-images 100

# Step 2: Annotate images and place labels in training/data/<task>/raw-data/

# Step 3: Create dataset from raw annotations
uv run python -m training.cli create-data --augment --visualize

# Step 4: Train model
uv run python -m training.cli train --model-size m --num-epochs 100
```

## Configuration

Default training parameters are configured in `src/digitex/config/settings.py`:

```python
class TrainingSettings(BaseSettings):
    data_subdir: str = "page"
    model_type: str = "seg"
    model_size: str = "m"
    pretrained_model_path: Optional[str] = None
    num_epochs: int = 100
    image_size: int = 640
    batch_size: int = 4
    overlap_mask: bool = False
    patience: int = 50
    seed: int = 42
```

Override with environment variables:

```bash
export TRAIN_NUM_EPOCHS=200
export TRAIN_BATCH_SIZE=8
export TRAIN_MODEL_SIZE=m
```

## Data Format

### Raw Data

Raw annotations should be in YOLO polygon format. Each image has a corresponding `.txt` file:

```
class_id x1 y1 x2 y2 ... xn yn
```

Coordinates are normalized (0-1).

### Dataset Structure

After creation, the dataset follows YOLO format:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Troubleshooting

### CUDA Not Available

Ensure NVIDIA GPU with CUDA is installed. Check with:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Dataset Issues

- Verify raw-data structure matches expected format
- Check annotation coordinate values are 0-1 normalized
- Use `--visualize` flag to inspect dataset quality

### Training Convergence

- Increase `--patience` for longer training
- Reduce `--batch-size` if GPU memory issues occur
- Try larger model sizes (`--model-size m/l/x`) for better accuracy
