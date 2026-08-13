"""Undo for the review window: a timeline of snapshots.

The window edits its regions in place — dragging a vertex rewrites a polygon,
relabelling rewrites a field — so undo cannot be a stack of inverse operations.
It is a stack of copies instead. A page carries tens of polygons of a few
points each, so a copy per edit costs nothing next to re-rendering the page.

The history owns its copying in both directions: what is pushed cannot be
changed from outside afterwards, and what is handed back is the caller's to
edit. Nothing here imports tkinter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from digitex.domain.entities import PixelPolygon
from digitex.extractors.placement import PageRegion

if TYPE_CHECKING:
    from collections.abc import Sequence

    from digitex.extractors.placement import PageExtractionState

# Enough to walk back through a page's whole edit session; a page nobody has
# spent 200 edits on does not need more, and the cap keeps a stuck key from
# growing the list without end.
DEFAULT_LIMIT = 200


def copy_regions(regions: Sequence[PageRegion]) -> list[PageRegion]:
    """Copy regions deeply enough that editing one cannot reach another."""
    return [
        PageRegion(
            label=region.label,
            polygon=PixelPolygon(list(region.polygon)),
            reading=region.reading,
        )
        for region in regions
    ]


@dataclass(frozen=True)
class EditSnapshot:
    """Everything one undo step puts back."""

    regions: list[PageRegion]
    state: PageExtractionState
    selected: int | None = None


class EditHistory:
    """A page's edit timeline, with a cursor to move along it.

    ``reset`` seeds it with the page as detected; every completed edit
    ``push``es the result. Undo therefore steps back onto the previous
    *result*, which is what the user last saw.
    """

    def __init__(self, limit: int = DEFAULT_LIMIT) -> None:
        self._limit = max(1, limit)
        self._timeline: list[EditSnapshot] = []
        self._cursor = -1

    @staticmethod
    def _snapshot(
        regions: Sequence[PageRegion],
        state: PageExtractionState,
        selected: int | None,
    ) -> EditSnapshot:
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

        del self._timeline[self._cursor + 1 :]
        self._timeline.append(self._snapshot(regions, state, selected))

        if len(self._timeline) > self._limit:
            del self._timeline[0 : len(self._timeline) - self._limit]
        self._cursor = len(self._timeline) - 1

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return -1 < self._cursor < len(self._timeline) - 1

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
        snapshot = self._timeline[self._cursor]
        return self._snapshot(snapshot.regions, snapshot.state, snapshot.selected)
