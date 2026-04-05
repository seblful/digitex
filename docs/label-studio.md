# Label Studio

Annotation platform for polygon segmentation labels.

**Note:** Run all commands from the project root directory.

## Start Server

```bash
uv run --env-file .env --with label-studio label-studio start
```

Server runs at `http://localhost:8080`.

## Configuration

Add to `.env`:

```
LABEL_STUDIO_API_KEY=your-api-key
LABEL_STUDIO_URL=http://localhost:8080
```

Get your API key from **Label Studio > Account & Settings > Access Token**.

## Data Flow

```
training/data/<task>/
├── annotations.json    # Label Studio export (polygon annotations)
├── images/             # Source page images
└── dataset/            # Created by dataset creator
    ├── train/
    ├── val/
    ├── test/
    └── data.yaml
```

Export annotations from Label Studio as JSON. The dataset creator reads `annotations.json` and `images/` to produce YOLO-format labels.

## Notes

Images are referenced in Label Studio as `data/page/images/<filename>.jpg` via local file storage.

## Auto-Prediction

Run a trained model on unannotated tasks and upload predictions back to Label Studio:

```bash
uv run python -m training.cli ls-predict --project-id 1 --model-path extraction/models/page.pt
```

**How it works:**

1. Fetches all tasks where `is_labeled=False`
2. Reads the image from local disk (via task URI)
3. Runs YOLO segmentation model
4. Uploads polygon predictions immediately per task
5. Skips tasks with missing files or failed predictions

**Requirements:**

- Label Studio running at `localhost:8080`
- `LABEL_STUDIO_API_KEY` set in `.env`
- Trained model `.pt` file
