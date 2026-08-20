"""Undo for the review window: a timeline of snapshots.

The window edits its regions in place — a drag rewrites a polygon, relabelling
rewrites a field — so undo cannot be a stack of inverse operations to replay
backwards. It is a stack of copies instead. A page holds tens of polygons of a
few points each, so one copy per edit costs nothing next to redrawing the page
that follows it.

The copying happens in both directions, and that is the whole correctness of the
thing: what was pushed cannot be changed from outside afterwards, and what comes
back is the caller's to edit. A shared polygon would make undo look like it
worked and then quietly move again with the next drag.

Nothing here imports tkinter, and nothing here knows what an edit was — only
that the page looked like this before it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from digitex.domain.placement import copy_regions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from digitex.domain.placement import PageExtractionState, PageRegion

# Enough to walk back through a page's whole edit session. Nobody spends 200
# edits on one page, and the cap is what stops a stuck key growing the list
# without end.
DEFAULT_LIMIT = 200


@dataclass(frozen=True)
class EditSnapshot:
    """Everything one undo step puts back."""

    regions: list[PageRegion]
    state: PageExtractionState
    selected: int | None = None


class EditHistory:
    """A page's edit timeline, with a cursor to move along it.

    ``reset`` seeds it with the page as detected; every completed edit ``push``es
    the result. So undo steps back onto the previous *result* — what the user
    last saw — rather than onto some half-finished intermediate.
    """

    def __init__(self, limit: int = DEFAULT_LIMIT) -> None:
        self._limit = max(1, limit)
        self._timeline: list[EditSnapshot] = []
        # -1 means "no timeline yet", which only `reset` and the first `push`
        # ever see.
        self._cursor = -1

    @staticmethod
    def _snapshot(
        regions: Sequence[PageRegion],
        state: PageExtractionState,
        selected: int | None,
    ) -> EditSnapshot:
        """Copy a page deeply enough that the timeline owns it alone."""
        return EditSnapshot(
            regions=copy_regions(regions),
            state=replace(state),
            selected=selected,
        )

    def reset(
        self,
        regions: Sequence[PageRegion],
        state: PageExtractionState,
        selected: int | None = None,
    ) -> None:
        """Start a fresh timeline at *regions* — a new page, nothing to undo."""
        self._timeline = [self._snapshot(regions, state, selected)]
        self._cursor = 0

    def push(
        self,
        regions: Sequence[PageRegion],
        state: PageExtractionState,
        selected: int | None = None,
    ) -> None:
        """Record the result of one edit, dropping any undone future."""
        if self._cursor < 0:
            self.reset(regions, state, selected)
            return

        # Editing after an undo abandons what was undone: the timeline is one
        # line, not a tree.
        del self._timeline[self._cursor + 1 :]
        self._timeline.append(self._snapshot(regions, state, selected))
        del self._timeline[: -self._limit]
        self._cursor = len(self._timeline) - 1

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._cursor < len(self._timeline) - 1

    def undo(self) -> EditSnapshot | None:
        """Step back one edit, or None at the start of the timeline."""
        if not self.can_undo:
            return None
        self._cursor -= 1
        return self._current()

    def redo(self) -> EditSnapshot | None:
        """Step forward one edit, or None at the end of the timeline."""
        if not self.can_redo:
            return None
        self._cursor += 1
        return self._current()

    def _current(self) -> EditSnapshot:
        """A copy of where the cursor sits, so the caller can edit it freely."""
        at = self._timeline[self._cursor]
        return self._snapshot(at.regions, at.state, at.selected)
