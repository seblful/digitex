# Extraction Guide

Extract question images from test books using YOLO segmentation.

## Quick Start

```bash
# Extract questions from a specific subject
digitex-extract extract-questions biology

# Same, checking every page in a window before its crops are saved
digitex-extract extract-questions biology --review

# Extract answers for a subject
digitex-extract extract-answers biology
```

Counting what came out and checking it against `answers.json` are both in the
review window's second tab — see [below](#the-review-window---review). There is
no renumbering command either: the window refuses numbering that would leave a
gap, so there is nothing to repair afterwards.

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

1. Reads images from `var/books/{subject}/images/{year}/`
1. Uses YOLO model to detect questions, options, and parts
1. Reads the option number and part letter off their markers with OCR
1. Numbers each question from those markers, continuing across pages
1. Crops and saves to `var/extraction/output/{subject}/{year}/{option}/{part}/`
1. Tracks progress in `var/extraction/progress.json`

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
| Nudge one | select it, then `←` `→` `↑` `↓` — `Shift` for 10px steps |
| Add / remove a point | right-click → *Insert point here* / *Delete point* |
| Draw a missing region | *Draw: question / option / part*, then drag a box |
| Delete a region | select it, press `Del` |
| Undo / redo | `Ctrl+Z` / `Ctrl+Y`, or the toolbar arrows |
| Fix a misread marker | right-click → *Set option number…* / *Set part*, or double-click its row |
| Relabel a region | right-click → *Label*, or press `1` `2` `3` (question / option / part) |
| Fix reading order | select a row, then `↑` `↓`, or *Sort by position* (`s`) |
| Move where the page starts | edit *Page starts at* — option, part, questions done |
| Zoom | wheel (at the cursor), `+` `-` `Fit` `1:1`, or `f` to fit |
| Fill the view with one region | right-click → *Zoom to region*, or `z` |
| Pan | middle-drag, the scrollbars, or the arrow keys with nothing selected |
| Step through the regions | `Tab` / `Shift+Tab` |

Numbering updates live as you edit, and is computed by the same code that
writes the files — the preview cannot disagree with what lands on disk. The
same goes for **Crop preview** under the region list: it is the extractor's own
cropping pipeline run on the selected question, so what it shows is the image
file that approving would write, deskewed and cleaned exactly as it will be
saved. Selecting a marker shows what OCR was pointed at instead.

The status bar under the page carries the run's position — which page of how
many, how many questions and markers are on it, and the range they would be
saved as.

**Numbering that would break the output tree is refused.** Every question's
file has to be the next one in its `{option}/{part}` folder. Land on a number
that already exists and it would overwrite an extracted question; land past the
end and the folder is left with a hole. Either way the offending question turns
red, *Approve & save* is disabled, and the message names the free number. Two
ways out:

- **Continue from disk** sets the entry counter so the page picks up right
  where the folder left off. Offered only when it would help — a page whose
  first question follows an option or part marker takes its numbering from that
  marker, not from the entry state.
- **Skip page**, when the page is simply already extracted.

This is why there is no renumbering command: gaps never get written.

The same rule blocks a question detected before any option/part marker has been
read — approve is disabled until you say where the page starts.

**Extracted so far** tab is the count and the answer check, per subject:
per-year option/part counts with anything off its year's mode in red, *Recount*
to refresh after approving pages, and *Check answers* to validate every
`answers.json` against the images on disk. It recounts when you open the tab
rather than on every page — walking the whole output tree per page would make
each one slower than the last.

Finally: **Approve & save** writes the crops (`Ctrl+Enter`), **Skip page**
writes nothing and leaves numbering where it was, **Abort run** stops
everything. Aborting leaves the year unfinished, so approved pages keep their
images and re-running continues where you left off.

One window serves the whole run — pages are loaded into it rather than each
opening its own — so the zoom, the pan, the window size and the tab you are on
carry from page to page. The zoom is only reset when a page turns up in a
different size from the one before it.

### `extract-answers`

Extract answer keys via OpenRouter vision API.

```bash
digitex-extract extract-answers <SUBJECT>
```

**Requirements:**

- Set `OPENROUTER_API_KEY` environment variable
- Answer images in `var/books/{subject}/answers/`
- Filename format: `YYYY_N.jpg` (e.g., `2016_1.jpg`)

## Directory Structure

Everything below lives under the data root — `var/` by default, or wherever
`PATH_DATA_ROOT` points. None of it is in version control.

```
var/
├── books/
│   └── {subject}/
│       ├── images/
│       │   ├── 2020/
│       │   │   ├── page1.jpg
│       │   │   └── page2.jpg
│       │   └── 2021/
│       └── answers/
│           ├── 2020_1.jpg
│           └── 2020_2.jpg
├── extraction/
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
EXTRACTION_IMAGE_FORMAT=jpg
EXTRACTION_QUESTION_MAX_WIDTH=2000
EXTRACTION_QUESTION_MAX_HEIGHT=2000

# Every data path derives from this one, which defaults to ./var
PATH_DATA_ROOT=/path/to/corpus

# OpenRouter (for answers)
OPENROUTER_API_KEY=your_api_key
OPENROUTER_MODEL=moonshotai/kimi-k2.6
```

The model is always `{PATH_DATA_ROOT}/models/page.pt` — a computed path with no
environment variable of its own. Pointing `PATH_DATA_ROOT` at a scratch corpus
is how you try a run without touching the real tree; it moves the books, the
model and the output together.

`PATH_DATA_ROOT` is resolved against the working directory when relative, so
run these commands from the repo root or set it to an absolute path. Nothing is
derived from where the package itself is installed.

A checkpoint carries the paths of the run that trained it, so one trained on
Linux used to fail to load on Windows with `cannot instantiate 'PosixPath'`
before a single page was read. Those paths mean nothing to inference, so the
predictor now maps them onto the local kind while the model loads — a model
trains on either platform and runs on both.

## Progress Tracking

Progress is automatically tracked in `var/extraction/progress.json`:

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
| No images folder | Create `var/books/{subject}/images/` |
| API key not set | Set `OPENROUTER_API_KEY` environment variable |
| Model not found | Put it at `{PATH_DATA_ROOT}/models/page.pt` |

## Best Practices

1. **Extract with `--review`** on a subject the model has not been trained on;
   the run is only as good as its worst page, and a misread option marker
   re-files every question after it
1. **Check progress** before re-running extraction
1. **Validate answers** in the review window's second tab after extraction
1. **Backup data** before bulk operations
1. **Use subject filtering** to process one subject at a time
