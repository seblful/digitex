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
# Count all subjects
uv run python -m training.cli count

# Count total images per subject
uv run python -m training.cli count --subject biology

# Count specific year/option/part
uv run python -m training.cli count --subject biology --year 2016 --option 1 --part A
```

### Renumber Files

```bash
# Preview renumbering from question 10
uv run python -m training.cli renumber --subject biology --year 2016 --option 1 --part A --start 10 --dry-run

# Apply renumbering
uv run python -m training.cli renumber --subject biology --year 2016 --option 1 --part A --start 10
```

### Add Manual Images

```bash
# Process all manual images
uv run python -m training.cli extract-manual

# Dry run (preview only)
uv run python -m training.cli extract-manual --dry-run
```

## Workflow

### 1. Count Existing Images

```bash
uv run python -m training.cli count --subject biology
```

### 2. Clean Broken Images

Remove corrupted files before adding new ones:

```bash
# Manually inspect and delete broken files
del extraction\output\biology\2016\1\A\broken_image.jpg
```

### 3. Prepare Images

Place manually cropped question images in `extraction/data/<subject>/`.

### 4. Renumber Files

If inserting a question that already exists, renumber first to avoid conflicts:

```bash
uv run python -m training.cli renumber --subject biology --year 2016 --option 1 --part A --start 10
```

### 5. Add Manual Images

```bash
uv run python -m training.cli extract-manual
```

### 6. Count Existing Images

```bash
uv run python -m training.cli count --subject biology
```

**Behavior:**

- Parses filename to extract year, option, part, question number
- Applies preprocessing (resize, segment enhancement)
- Saves to `extraction/output/<subject>/<year>/<option>/<part>/<question>.jpg`
- Automatically renumbers subsequent files if target exists
- Deletes source manual file after processing

## Training Data Paths

For training workflows, list image paths in `training/data/<task>/images.txt`:

```
books/biology/2016/14.jpg
books/biology/2016/15.jpg
books/physics/2024/3.jpg
```

**Format:**

- One relative path per line
- Paths are relative to project root
- Structure: `books/<subject>/<year>/<page>.jpg`

Then add them with:

```bash
uv run python -m training.cli add-images page
```

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
