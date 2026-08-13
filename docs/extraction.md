# Extraction Guide

Extract question images from test books using YOLO segmentation.

## Quick Start

```bash
# Extract questions from a specific subject
digitex-extract extract-questions biology

# Same, checking every page in a window before its crops are saved
digitex-extract extract-questions biology --review

# Count extracted questions for a subject
digitex-extract count-questions biology

# Fix numbering gaps for a subject
digitex-extract renumber-questions biology

# Extract answers for a subject
digitex-extract extract-answers biology

# Check answers for a subject
digitex-extract check-answers biology
```

## Commands

### `extract-questions`

Extract question images from a specific subject.

```bash
digitex-extract extract-questions <SUBJECT> [--review]
```

**Arguments:**

- `<SUBJECT>` - Subject name (e.g., `biology`, `chemistry`)

**Options:**

- `--review` - Check every page in a window before its crops are saved

**Process:**

1. Reads images from `books/{subject}/images/{year}/`
1. Uses YOLO model to detect questions, options, and parts
1. Reads the option number and part letter off their markers with OCR
1. Numbers each question from those markers, continuing across pages
1. Crops and saves to `extraction/data/output/{subject}/{year}/{option}/{part}/`
1. Tracks progress in `extraction/data/progress.json`

### The review window (`--review`)

Every step above is a guess that can go wrong silently — a polygon that clips
the question, a misread option number that re-files the rest of the book. With
`--review`, each page stops in a window showing the page, its detected polygons
and the `{option}/{part}/{number}` every question would be saved as, and
nothing is written until you approve it.

**This page** tab:

| Do this | Like this |
| :------ | :-------- |
| Move a polygon | drag inside it |
| Reshape one | drag a white handle (selected polygon only) |
| Add / remove a point | right-click → *Insert point here* / *Delete point* |
| Draw a missing region | *Draw: question / option / part*, then drag a box |
| Delete a region | select it, press `Del` |
| Fix a misread marker | right-click → *Set option number…* / *Set part* |
| Relabel a region | right-click → *Label* |
| Fix reading order | select a row, then `↑` `↓`, or *Sort by position* |
| Move where the page starts | edit *Page starts at* — option, part, questions done |
| Zoom | mouse wheel, or `-` `+` `Fit` |

Numbering updates live as you edit, and is computed by the same code that
writes the files — the preview cannot disagree with what lands on disk. A page
that starts with a question before any option/part marker is reported in red
and cannot be approved until you say where the page starts.

**Extracted so far** tab carries what `count-questions` and `check-answers`
print: per-year option/part counts, with anything off its year's mode in red,
and the answers.json check on demand.

Finally: **Approve & save** writes the crops (`Ctrl+Enter`), **Skip page**
writes nothing and leaves numbering where it was, **Abort run** stops
everything. Pages already approved keep their images and the year is not marked
complete, so re-running continues where you left off.

### `count-questions`

Count extracted images by year/option/part for a specific subject.

```bash
digitex-extract count-questions <SUBJECT>
```

**Arguments:**

- `<SUBJECT>` - Subject name (e.g., `biology`, `chemistry`)

### `renumber-questions`

Renumber images to fill gaps (e.g., 1,2,4,5 → 1,2,3,4) for a specific subject.

```bash
digitex-extract renumber-questions <SUBJECT> [--dry-run]
```

**Arguments:**

- `<SUBJECT>` - Subject name (e.g., `biology`, `chemistry`)

**Options:**

- `--dry-run` - Preview changes without applying (default: true)

### `extract-answers`

Extract answer keys via OpenRouter vision API.

```bash
digitex-extract extract-answers <SUBJECT>
```

**Requirements:**

- Set `OPENROUTER_API_KEY` environment variable
- Answer images in `books/{subject}/answers/`
- Filename format: `YYYY_N.jpg` (e.g., `2016_1.jpg`)

### `check-answers`

Validate answers.json against extracted images.

```bash
digitex-extract check-answers <SUBJECT>
```

**Checks:**

- Each year has answers.json
- Questions match between images and answers
- All options have same questions

## Directory Structure

```
books/
└── {subject}/
    ├── images/
    │   ├── 2020/
    │   │   ├── page1.jpg
    │   │   └── page2.jpg
    │   └── 2021/
    └── answers/
        ├── 2020_1.jpg
        └── 2020_2.jpg

extraction/
├── data/
│   ├── progress.json
│   └── output/
│       └── {subject}/
│           └── {year}/
│               ├── answers.json
│               ├── 1/
│               │   ├── A/
│               │   │   ├── 1.jpg
│               │   │   └── 2.jpg
│               │   └── B/
│               └── 2/
└── models/
    └── page.pt
```

## Configuration

Set environment variables or use `.env`:

```bash
# Extraction settings
EXTRACTION_MODEL_PATH=extraction/models/page.pt
EXTRACTION_IMAGE_FORMAT=jpg
EXTRACTION_QUESTION_MAX_WIDTH=2000
EXTRACTION_QUESTION_MAX_HEIGHT=2000

# OpenRouter (for answers)
OPENROUTER_API_KEY=your_api_key
OPENROUTER_MODEL=moonshotai/kimi-k2.6
```

## Progress Tracking

Progress is automatically tracked in `extraction/data/progress.json`:

```json
{
  "biology": ["2020", "2021"],
  "chemistry": ["2019", "2020"]
}
```

## Error Handling

Common errors and solutions:

| Error | Solution |
| ----------------- | ------------------------------------------ |
| Subject not found | Check subject name matches folder |
| No images folder | Create `books/{subject}/images/` |
| API key not set | Set `OPENROUTER_API_KEY` environment variable |
| Model not found | Check `EXTRACTION_MODEL_PATH` |

## Best Practices

1. **Always use `--dry-run`** first with `renumber-questions`
1. **Extract with `--review`** on a subject the model has not been trained on;
   the run is only as good as its worst page, and a misread option marker
   re-files every question after it
1. **Check progress** before re-running extraction
1. **Validate answers** with `check-answers` after extraction
1. **Backup data** before bulk operations
1. **Use subject filtering** to process one subject at a time
