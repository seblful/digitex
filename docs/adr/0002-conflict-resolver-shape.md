# ADR 0002 — Conflict resolution is a callable, not a Protocol

**Status:** Accepted
**Date:** 2026-05-21

## Context

`PageExtractor` may discover that the file it is about to write already
exists — typically because the Option counter advanced incorrectly on a
prior page. Originally this was modeled as a `ConflictResolutionStrategy`
`Protocol` with two implementations: `AutoConflictResolution` (used by
every production code path) and `InteractiveConflictResolution` (used only
in notebook contexts that nothing in the repo currently exercises).

The Protocol carried 12 lines of interface for one keyword-argument call
returning `int`. Apply the deletion test: deleting the Protocol concentrates
zero complexity at the call site — `PageExtractor` has one such site.
*One adapter in real use = hypothetical seam.*

## Decision

Replace the Protocol with:

```python
ConflictResolver = Callable[[Conflict], int]
```

where `Conflict` is a small dataclass carrying `new_image`, `existing_path`,
`current_option`, and `source_image_name`. Two module-level functions —
`keep_current_option` (default) and `prompt_user` — satisfy the alias.

`PageExtractor.__init__` takes `on_conflict: ConflictResolver | None = None`
and defaults to `keep_current_option`. The factory passes `on_conflict`
through unchanged.

## Consequences

- The seam is now a one-line type alias; no Protocol class, no abstract
  method.
- Adding a new resolver is one function, not a class with one method.
- A caller wanting a parameterized resolver (e.g. a custom prompter) uses
  `functools.partial(prompt_user, prompter=my_prompter)` instead of
  subclassing.
- `Conflict` is a dataclass, so resolvers can pattern-match its fields if
  needed; callers can mock by passing any callable.

## Why this is not a regression

The previous Protocol was inviting a third implementation that never
arrived. If a future caller actually needs strategy-object semantics
(stateful resolvers with setup/teardown), the alias can be expanded back
into a Protocol — but only when the second real adapter shows up. Until
then the callable is the right shape.

## Status of `InteractiveConflictResolution`

The interactive resolver survives as `prompt_user` for notebook use. If it
gains no callers within the codebase by the next architecture pass, it is a
deletion candidate too.
