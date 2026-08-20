"""Tests for the review seam — what a reviewer is handed, and its default.

The numbering rule the seam enforces moved to `test_domain_numbering`; what is
left here is the shape of the hand-off itself.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from digitex.domain.entities import PixelPolygon
from digitex.domain.placement import PageExtractionState, PageRegion
from digitex.pipeline.review import PageProposal, ReviewedPage, accept_page

POLYGON = PixelPolygon([(10, 40), (200, 40), (200, 80), (10, 80)])


class TestAcceptPage:
    def test_the_default_reviewer_hands_back_what_it_was_given(self) -> None:
        """Extraction without a reviewer must behave as it did before there was one."""
        regions = [PageRegion(label="question", polygon=POLYGON)]
        state = PageExtractionState(option=1, part="A")
        proposal = PageProposal(
            image=Image.new("RGB", (1, 1)),
            regions=regions,
            state=state,
            output_dir=Path("out"),
        )

        reviewed = accept_page(proposal)

        assert reviewed == ReviewedPage(regions=regions, state=state)
