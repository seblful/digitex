# Label Studio

Annotation platform for polygon segmentation labels.

For the full training pipeline and dataset workflow, see [Training](training.md).

**Note:** Run all commands from the project root directory.

## Start Server

```bash
uv run --env-file .env label-studio start
```

Server runs at `http://localhost:8080`.

## Configuration

Add to `.env`:

```
LABEL_STUDIO_API_KEY=your-api-key
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:/path/to/digitex
```

Get your API key from **Label Studio > Account & Settings > Access Token**.

The document root is the only directory the server will serve an image from, and
every task path is relative to it. Point it at the project root, not at the
image directory — the tasks name the rest of the path themselves.

## Data Flow

Annotations are stored in `var/training/data/<task>/` (see [Training](training.md) for full directory structure).

Export annotations from Label Studio as JSON to `annotations.json` in the task directory.

## Notes

A local files storage (**Settings > Cloud Storage**) points at
`var/training/data/page/images`, and each task it syncs references its image
relative to the document root — `var/training/data/page/images/<filename>.jpg`.

A task synced from such a storage files the URI under `$undefined$` rather than
`image`: Label Studio names the column that when the import carries no field
name, and resolves it against the `$image` tag in the label config when it
renders the task. Read it with `digitex.domain.geometry.task_image_path`, which
takes either key.

## Repairing Moved Images

Moving the image pool strands every task synced before the move: its URI names
the old path, so the image 404s in the editor, and the next sync reads the same
files as new ones and imports a second, unannotated task for each.

Point the storage at the new directory, sync it, then move the annotations onto
the tasks that now hold their images:

```bash
uv run --env-file .env python tools/fix_task_paths.py fix-task-paths --project-id 1
```

It prints the plan and changes nothing. Add `--no-dry-run` to apply, which dumps
every task it is about to delete to `var/label-studio/stranded-tasks-*.json`
first. A moved annotation keeps its result, its labels, its annotator and its
lead time, but gets a new id and timestamp; predictions on a stranded task are
dropped rather than moved — rerun `ls-predict` if they are wanted.

## Auto-Prediction

Run a trained model on unannotated tasks and upload predictions back to Label Studio:

```bash
uv run digitex-train ls-predict --project-id 1 --model-path var/models/page.pt
```

**How it works:**

1. Fetches all tasks where `is_labeled=False` and no prediction exists yet
1. Reads the image from local disk (via task URI)
1. Runs YOLO segmentation model
1. Uploads polygon predictions immediately per task
1. Skips tasks with missing files or failed predictions

Each region carries the model's confidence, and each task the mean of its
regions — sort the Data Manager by score to review the shakiest pages first.

**Requirements:**

- Label Studio running at `localhost:8080`
- `LABEL_STUDIO_API_KEY` set in `.env`
- Trained model `.pt` file (see [Training](training.md) for training workflow)
