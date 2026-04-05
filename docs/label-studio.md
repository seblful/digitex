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

## Upload Labels

Upload YOLO polygon annotations as pre-annotated tasks:

```bash
uv run python training/scripts/upload_labels.py --project-id 1
```

**Options:**

- `--project-id`: Label Studio project ID (required)
- `--batch-size`: Tasks per API call (default: 50)

## Rename Labels

Strip hex prefix and rename to structured format:

```bash
uv run python training/scripts/rename_labels.py
```

Format: `biology_{year}_{page}_{type}` — e.g. `biology_2016_12_medium`

## Data Flow

```
training/data/page/raw-data/
├── images/    # Source images
└── labels/    # YOLO polygon format
```

Images are referenced in Label Studio as `data/page/images/<filename>.jpg` via local file storage.
