# ADR 0001 — Settings are injected at module boundaries

**Status:** Accepted
**Date:** 2026-05-21

## Context

`get_settings()` is a `lru_cache`'d singleton over the `Settings` tree. It is
the highest-betweenness node in the codebase (graph betweenness 0.336) — it
touches CLI commands, the bot, the extractor factory, the trainer, the
predictor, and several utility modules.

Two patterns were in tension:

1. **Inject** — pass `Settings` (or the relevant sub-tree) explicitly into
   constructors so callers see the dependency in the type signature.
1. **Lazy-load** — call `get_settings()` directly inside any function that
   needs config.

The recent refactor (`e77106e`) moved `ExtractorFactory` to pattern (1).

## Decision

Settings are **injected at the outermost module boundary** the dependency
crosses:

- **CLI entry points** (`cli/extraction.py`, `cli/training.py`, `cli/bot.py`)
  call `get_settings()` and pass the resolved `Settings` instance into the
  factory or service they construct.
- **Factories and services** (`ExtractorFactory`, training entry points)
  take `Settings` as a constructor argument and read sub-settings off of
  `self._settings` — they never call `get_settings()` themselves.
- **Leaf utilities** that need a single piece of config (`logging.setup`,
  `utils.get_tz`) MAY call `get_settings()` directly. The criterion is:
  *would adding a `Settings` parameter to this function force every caller
  to also take one?* If yes, lazy-load is acceptable.

Constructors that take `Settings` must not also accept individual overrides
that duplicate `Settings` fields — pick one shape per parameter. The factory
exception (it accepts optional `model_path`, `image_format`, … overrides) is
load-bearing for CLI flags and stays.

## Consequences

- The type of a constructor argument tells you whether a module depends on
  config. Greppable.
- Tests construct fake `Settings` (Pydantic) instead of monkey-patching
  `get_settings`.
- Adding a new `Settings` field doesn't require auditing every call site of
  `get_settings()` — only modules that already inject it.
- `get_settings()` itself remains. Removing the singleton would force
  `Settings` plumbing through every leaf utility, which is the larger evil.

## Non-goals

This ADR does not specify whether sub-settings (`PathsSettings`,
`ExtractionSettings`, …) should be split out and injected separately rather
than passing the whole `Settings`. That decision is local to each module;
the factory currently takes the whole tree, which keeps the constructor
stable as new sub-settings appear.
