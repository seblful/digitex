# Architecture Decision Records

Lightweight records of architecture decisions that the codebase wouldn't
otherwise speak for itself.

ADRs explain *why* a design is the way it is so future explorers
(human or AI) don't re-suggest a refactor that has already been considered
and rejected for a reason that isn't visible in the code.

## Index

- [ADR 0001](0001-settings-injection.md) — Settings are injected at module boundaries
- [ADR 0002](0002-conflict-resolver-shape.md) — Conflict resolution is a callable, not a Protocol
- [ADR 0003](0003-bot-answer-flow-seam.md) — The bot's "render a Question + cache file_id" recipe lives in `answer_flow`

## When to write an ADR

When a load-bearing decision is hard to recover from the code:

- "Why is this a callable and not a class?" — yes, ADR.
- "Why does this constructor take `Settings` and not `PathsSettings`?" — yes,
  ADR.
- "What's the deletion criterion for this Protocol?" — yes, ADR.
- "Why is this function called `ask_question`?" — usually no; put it in
  CONTEXT.md if it's about naming.

If a decision is self-evident from the diff and the commit message, skip
the ADR.
