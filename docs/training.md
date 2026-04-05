# Training

YOLO-based training pipeline for document segmentation.

For Label Studio setup, configuration, and server management, see [Label Studio](label-studio.md).

**Note:** Run all commands from the project root directory.

## Directory Structure

```
training/
├── cli.py              # Unified CLI for training workflows
├── data/               # Training data (gitignored)
│   └── <task>/         # Task-specific data (page, question, part)
│       ├── annotations.json  # Label Studio export
│       ├── images/           # Source page images
│       ├── dataset/          # Processed YOLO dataset
│       └── check-images/     # Visual verification images
└── runs/               # Ultralytics runs (gitignored)
```

## Commands

### 1. Select Random Pages

Select random images from book folders and save them for annotation:

```bash
uv run python -m training.cli select-random-pages --data-subdir page --num-images 100
```

**Options:**
- `--data-subdir`: Task type (default: `page`)
- `--num-images`: Number of images to select (default: 100)

**Requirements:**
- Book images in `books/<subject>/images/<year>/` directory

### 2. Create Dataset

Convert annotations from Label Studio to YOLO dataset format:

```bash
uv run python -m training.cli create-dataset --data-subdir page --train-split 0.8
```

**Options:**
- `--data-subdir`: Task type (default: `page`)
- `--train-split`: Training/validation split ratio (default: 0.8)
- `--vis-images`: Number of images to visualize (default: 20)
- `--augment`: Enable data augmentation (default: False)
- `--aug-images`: Number of augmented images (default: 100)

**Requirements:**
- `annotations.json` in `training/data/<task>/`
- Images in `training/data/<task>/images/`

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

### 4. Predict Label Studio Tasks

Run trained model on unannotated Label Studio tasks and upload predictions (see [Label Studio](label-studio.md) for setup):

```bash
uv run python -m training.cli ls-predict --project-id 1 --model-path training/data/page/models/best.pt
```

**Options:**
- `--project-id`: Label Studio project ID (required)
- `--model-path`: Path to trained model (default from `TRAIN_PRETRAINED_MODEL_PATH`)

**Behavior:**
- Iterates tasks where `is_labeled=False`
- Skips tasks with missing images (logs warning)
- Uploads prediction immediately after each task
- Safely re-runnable — completed tasks are skipped

## Complete Workflow

Run from the project root directory:

```bash
# Step 1: Select random pages from books
uv run python -m training.cli select-random-pages --num-images 100

# Step 2: Annotate images in Label Studio (see [Label Studio](label-studio.md)), then export annotations.json

# Step 3: Create dataset from annotations (visualizes automatically)
uv run python -m training.cli create-dataset --augment

# Step 4: Train model
uv run python -m training.cli train --model-size m --num-epochs 100

# Step 5: Predict unannotated tasks in Label Studio
uv run python -m training.cli ls-predict --project-id 1 --model-path training/data/page/models/best.pt
```

## Configuration

Default training parameters are configured in `src/digitex/config/settings.py`:

```python
class TrainingSettings(BaseSettings):
    model_type: str = "seg"
    model_size: str = "m"
    num_epochs: int = 100
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

### Annotations (Label Studio export)

`annotations.json` contains polygon annotations exported from Label Studio. Each entry has:
- `image`: URI referencing the image file
- `label`: list of polygons with percentage coordinates (0-100) and class labels

The dataset creator converts percentage coordinates to YOLO normalized format (0-1) automatically.

### Dataset Structure

After creation, the dataset follows YOLO format:

```
dataset/
├── train/
│   ├── *.jpg
│   └── *.txt        # YOLO polygon labels
├── val/
├── test/
└── data.yaml
```

## Troubleshooting

### CUDA Not Available

Ensure NVIDIA GPU with CUDA is installed. Check with:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Dataset Issues

- Check annotation coordinate values in `annotations.json`
- Use `--visualize` flag to inspect dataset quality

### Training Convergence

- Increase `--patience` for longer training
- Reduce `--batch-size` if GPU memory issues occur
- Try larger model sizes (`--model-size m/l/x`) for better accuracy
