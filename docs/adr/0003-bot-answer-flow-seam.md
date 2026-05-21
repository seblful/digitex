# ADR 0003 — The bot's "render a Question + cache file_id" recipe lives in `answer_flow`

**Status:** Accepted
**Date:** 2026-05-21

## Context

Both `bot.handlers.testing` (Standard testing) and `bot.handlers.random`
(Random testing) need to do the same thing when presenting a Question:

1. Send the Question's image to the chat. Part A goes with the option-picker
   keyboard; Part B is followed by an "enter your answer" prompt.
1. If the image was uploaded fresh, cache the new Telegram `file_id` so
   future renders skip the upload.

Pre-refactor, this shape was reimplemented in both handler files — about
~20 lines each, with subtle drift between them (different keyword argument
orderings, slightly different caching paths).

`bot.renderer.send_question` is the lower-level primitive — it sends the
image and returns the new `file_id` (or `None`). It does not know about
keyboards, prompts, or caching.

## Decision

A new module `digitex.bot.answer_flow` owns the **recipe**:

```python
async def ask_question(
    bot: Bot,
    message: types.Message,
    question: Question,
    pool: AsyncConnectionPool,
    *,
    caption: str | None = None,
    parse_mode: str | None = None,
) -> None:
```

`ask_question` chooses the keyboard / follow-up message based on
`question.part`, calls `send_question`, and caches the `file_id` via a UoW.

Handler modules call `ask_question` and keep their own FSM transitions.

## Consequences

- Adding a third bot conversation shape (e.g. timed exam) does not require
  copy-pasting the render recipe — it calls `ask_question` and owns only
  its own FSM logic.
- `bot.renderer` stays pure (no DB dependency); `bot.answer_flow` is where
  the DB and the renderer meet.
- The seam is *named* — `ask_question` is the verb for "present a Question
  to the chat", per CONTEXT.md.

## What this seam is not

`ask_question` deliberately does NOT own the "record an answer" half of
the loop, because Standard testing records to a Session and Random testing
does not — those flows diverge before the recording step. If a future
mode unifies them, that recipe can join `answer_flow` next to
`ask_question`.
