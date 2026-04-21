# Extraction Guide

Extract question images from test books using YOLO segmentation.

## Quick Start

```bash
# Extract questions from a specific subject
digitex-extract extract-questions biology

# Count extracted questions for a subject
digitex-extract count-questions biology

# Fix numbering gaps for a subject
digitex-extract renumber-questions biology

# Add manually cropped questions for a subject
digitex-extract add-questions-manually biology

# Extract answers for a subject
digitex-extract extract-answers biology

# Check answers for a subject
digitex-extract check-answers biology
```

## Commands

### `extract-questions`

Extract question images from a specific subject.

```bash
digitex-extract extract-questions <SUBJECT>
```

**Arguments:**

- `<SUBJECT>` - Subject name (e.g., `biology`, `chemistry`)

**Process:**

1. Reads images from `books/{subject}/images/{year}/`
2. Uses YOLO model to detect questions, options, and parts
3. Crops and saves to `extraction/data/output/{subject}/{year}/{option}/{part}/`
4. Tracks progress in `extraction/data/progress.json`

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

### `add-questions-manually`

Integrate manually cropped question images for a specific subject.

```bash
digitex-extract add-questions-manually <SUBJECT> [--dry-run]
```

**Arguments:**
- `<SUBJECT>` - Subject name (e.g., `biology`, `chemistry`)

**Manual Image Format:**
- Place in: `extraction/data/manual/{subject}/`
- Filename: `YYYY_OPTION_PART_QUESTION.png`
- Example: `biology/2016_3_A_20.png`

**Options:**
- `--dry-run` - Preview changes without applying

### `extract-answers`

Extract answer keys using Mistral OCR.

```bash
digitex-extract extract-answers <SUBJECT>
```

**Requirements:**

- Set `MISTRAL_API_KEY` environment variable
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
│   ├── manual/
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

# Mistral OCR (for answers)
MISTRAL_API_KEY=your_api_key
MISTRAL_OCR_MODEL=mistral-ocr-latest
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

| Error             | Solution                                   |
| ----------------- | ------------------------------------------ |
| Subject not found | Check subject name matches folder          |
| No images folder  | Create `books/{subject}/images/`           |
| API key not set   | Set `MISTRAL_API_KEY` environment variable |
| Model not found   | Check `EXTRACTION_MODEL_PATH`              |

## Best Practices

1. **Always use `--dry-run`** first with renumber/manual commands
2. **Check progress** before re-running extraction
3. **Validate answers** with `check-answers` after extraction
4. **Backup data** before bulk operations
5. **Use subject filtering** to process one subject at a time
