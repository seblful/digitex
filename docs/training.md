# Training

YOLO-based training pipeline for document segmentation.

For Label Studio setup, configuration, and server management, see [Label Studio](label-studio.md).

**Note:** Run all commands from the project root directory.

## Directory Structure

```
training/
├── cli.py              # Unified CLI for training workflows
├── configs/            # Training configurations
│   └── page.yaml       # Page segmentation config
├── data/               # Training data (gitignored)
│   └── <task>/         # Task-specific data (page, question, part)
│       ├── annotations.json  # Label Studio export
│       ├── images/           # Source page images
│       └── dataset/          # Processed YOLO dataset
└── runs/               # Ultralytics runs (gitignored)
```

## Commands

### 1. Select Random Pages

Select random images from book folders and save them for annotation:

```bash
uv run python -m training.cli select-random-pages --num-images 100
```

**Options:**

- `--num-images`: Number of images to select (default: 100)

**Requirements:**

- Book images in `books/<subject>/images/<year>/` directory

### 2. Add Images from File

Add specific images listed in a text file to the training data:

```bash
uv run python -m training.cli add-images page
```

**Arguments:**

- `page`: Task type (required)

**Requirements:**

- `paths.txt` in `training/data/<task>/` with one relative path per line:

```
books/biology/images/2024/10.jpg
books/biology/images/2024/15.jpg
books/biology/images/2023/5.jpg
```

**Behavior:**

- Copies and resizes images to 640x640
- Renames to `{subject}_{year}_{page}.jpg`
- Skips images that already exist in the output directory
- Logs summary of processed/skipped/missing

### 3. Create Dataset

Convert annotations from Label Studio to YOLO dataset format:

```bash
uv run python -m training.cli create-dataset <task> --train-split 0.8
```

**Arguments:**

- `<task>`: Task type (e.g., `page`)

**Options:**

- `--train-split`: Training/validation split ratio (default: 0.8)

**Requirements:**

- `annotations.json` in `training/data/<task>/`
- Images in `training/data/<task>/images/`

### 4. Train Model

Train YOLO segmentation model using configuration from `training/configs/<config>.yaml`:

```bash
uv run python -m training.cli train --config page
```

**Options:**

- `--config`: Config name without `.yaml` extension (default: `page`)

**Configuration:**

All training parameters are managed in `training/configs/<config>.yaml`:
- `model`: Model architecture (e.g., `yolo26m-seg.yaml`)
- `data`: Path to dataset YAML (e.g., `training/data/page/dataset/data.yaml`)
- `epochs`: Number of training epochs
- `batch`: Batch size
- `imgsz`: Input image size
- `device`: Device for training (`-1` for auto-selection)
- `patience`: Early stopping patience
- `mask_ratio`: Mask downsample ratio
- `overlap_mask`: Mask overlap for segmentation
- `seed`: Random seed
- etc.

To modify training parameters, edit `training/configs/<config>.yaml`.

### 5. Predict Label Studio Tasks

Run trained model on unannotated Label Studio tasks and upload predictions (see [Label Studio](label-studio.md) for setup):

```bash
uv run python -m training.cli ls-predict --project-id 1 --model-path training/runs/train/weights/best.pt
```

**Options:**

- `--project-id`: Label Studio project ID (required)
- `--model-path`: Path to trained model (required)

**Behavior:**

- Iterates tasks where `is_labeled=False`
- Skips tasks with missing images (logs warning)
- Uploads prediction immediately after each task
- Safely re-runnable — completed tasks are skipped

## Complete Workflow

Run from the project root directory:

```bash
# Step 1: Add images for annotation
uv run python -m training.cli select-random-pages --num-images 100

# Step 2: Annotate images in Label Studio (see [Label Studio](label-studio.md)), then export annotations.json

# Step 3: Create dataset from annotations
uv run python -m training.cli create-dataset page

# Step 4: Train model (uses training/config.yaml)
uv run python -m training.cli train

# Step 5: Predict unannotated tasks in Label Studio
uv run python -m training.cli ls-predict --project-id 1 --model-path training/runs/train/weights/best.pt
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

### Training Convergence

- Increase `patience` in `training/config.yaml` for longer training
- Reduce `batch` in `training/config.yaml` if GPU memory issues occur
- Try larger model sizes by changing `model` in `training/config.yaml`
