"""Questions printed across a page break, and the piece one page owes the next.

A question that straddles the fold of a book is two pictures of one question:
the bottom of one page and the top of the next. The reviewer marks the first
picture as continuing (``PageRegion.joins_next``), which takes it out of the
numbering — nothing is written for it and it consumes no question number. Its
crop waits here instead, and the next page's first question is saved as the two
of them stacked.

"Piece", not "part": a part is what A and B are. This is one picture of a
question, not one of a paper's sections.

The carry is threaded through a book the way the numbering state is — one
object handed to every page, holding whatever the page before it could not
finish. Which is why it lives here rather than inside :class:`PageExtractor`: a
book's pages share one carry, and two books share nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

# The white band a piece sits below the one above it by default, before the
# reviewer lines the two up. Wide enough to read as a join rather than as the
# next line of the text.
PIECE_GAP = 16


@dataclass(frozen=True)
class HeldPiece:
    """One piece of a question, cut from a page that could not finish it.

    ``image`` is the crop as the extractor cuts it — masked, deskewed, and not
    yet capped to the question size, because the cap belongs to the whole
    question and the whole question is not here yet. ``page_name`` is the page
    it came off, so a reviewer looking at a carried piece can see where it is
    from. ``offset`` is how it was lined up against the piece above it, which
    only a question spanning three pages has.
    """

    image: Image.Image
    page_name: str
    offset: tuple[int, int] = (0, 0)


@dataclass
class PageCarry:
    """The question pieces a page held for the next page to finish.

    Empty for every page whose questions are all whole, which is nearly all of
    them. A piece still in here when a book ends was never joined, and the
    caller has to say so — nothing was written for it.
    """

    pieces: list[HeldPiece] = field(default_factory=list)

    def take(self) -> list[HeldPiece]:
        """Hand the held pieces to the page taking them on, leaving this empty."""
        taken = self.pieces
        self.pieces = []
        return taken

    def hold(self, pieces: list[HeldPiece]) -> None:
        """Keep *pieces* for the next page."""
        self.pieces = pieces
