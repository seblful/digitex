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
