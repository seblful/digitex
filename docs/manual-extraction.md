# Manual Extraction

Guide for manually adding extraction data, including counting, renumbering, and variable naming conventions.

## Directory Structure

```
extraction/
├── data/                 # Manual input images
│   └── <subject>/       # Subject folder (e.g., biology, physics)
│       └── *.png        # Manual images with naming convention
└── output/              # Processed output
    └── <subject>/
        └── <year>/
            └── <option>/
                └── <part>/
                    └── <question_number>.jpg
```

## Image Naming Convention

Manual images must follow this naming pattern:

```
YYYY_OPTION_PART_QUESTION.png
```

**Example:**

- `2016_3_A_20.png` - Year 2016, Option 3, Part A, Question 20
- `2024_1_A_5.png` - Year 2024, Option 1, Part A, Question 5

**Rules:**

- `YYYY`: 4-digit year
- `OPTION`: Option number (1-10)
- `PART`: Part letter (A or B)
- `QUESTION`: Question number (integer)

## CLI Commands

### Count Images

```bash
# Count images for a specific subject
digitex-extract count-questions biology
```

### Renumber Files

```bash
# Preview renumbering for a subject
digitex-extract renumber-questions biology --dry-run

# Apply renumbering
digitex-extract renumber-questions biology
```

### Add Manual Images

```bash
# Process all manual images for a subject
digitex-extract add-questions-manually biology

# Dry run (preview only)
digitex-extract add-questions-manually biology --dry-run
```

## Workflow

### 1. Count Existing Images

```bash
digitex-extract count-questions biology
```

### 2. Clean Broken Images

Remove corrupted files before adding new ones:

```bash
# Manually inspect and delete broken files
del extraction\output\biology\2016\1\A\broken_image.jpg
```

### 3. Prepare Images

Place manually cropped question images in `extraction/data/manual/<subject>/`.

**Important:** Manual images must be placed in the `manual` subdirectory, not directly in the subject folder.

### 4. Renumber Files

If inserting a question that already exists, renumber first to avoid conflicts:

```bash
digitex-extract renumber-questions biology --dry-run
```

### 5. Add Manual Images

```bash
digitex-extract add-questions-manually biology --dry-run  # Preview first
digitex-extract add-questions-manually biology            # Apply changes
```

### 6. Verify Results

```bash
digitex-extract count-questions biology
```

**Behavior:**

- Parses filename to extract year, option, part, question number
- Applies preprocessing (resize, segment enhancement)
- Saves to `extraction/output/<subject>/<year>/<option>/<part>/<question>.jpg`
- Automatically renumbers subsequent files if target exists
- Deletes source manual file after processing

## Directory Locations

**Manual Input:** `extraction/data/manual/<subject>/`
- Place your manually cropped `.png` files here
- Files are processed and moved to output
- Source files are deleted after processing

**Output:** `extraction/data/output/<subject>/<year>/<option>/<part>/`
- Processed images saved here
- Format: `<question_number>.jpg`

## Troubleshooting

### Filename Parse Errors

Ensure manual images follow exact naming convention: `YYYY_OPTION_PART_QUESTION.png`

Invalid examples:

- `2016_3_a_20.png` (lowercase part)
- `2016_3_A_.png` (missing question number)
- `biology_2016_1_A_5.png` (extra prefix)

### Duplicate Question Numbers

The extractor automatically shifts existing files when a duplicate is detected. Use `--dry-run` to preview changes before applying.

### Missing Images

Check that source images are in `extraction/data/<subject>/` and have `.png` extension.
