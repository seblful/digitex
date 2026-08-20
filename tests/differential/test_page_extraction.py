"""Extraction writes the same files it wrote when the recording was made.

This is the assertion the whole restructuring leans on. Every phase that moves
extraction code has to leave this test passing, and it compares the one thing
that matters to the corpus: the bytes of every question image, and where each
one landed in the output tree.

It is deliberately not a unit test. It runs the real page walk, the real
numbering, the real cropping, stacking and capping, over a real book —
substituting only the answers that come from outside the code, which the
recording holds.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from digitex.pipeline.book import BookExtractor
from digitex.pipeline.page import PageExtractor
from digitex.pipeline.recording import (
    ReplayPredictor,
    ReplayTextExtractor,
    directory_digests,
    replay_config,
)

if TYPE_CHECKING:
    from pathlib import Path

    from digitex.pipeline.base import ExtractionConfig
    from digitex.pipeline.recording import Recording

pytestmark = pytest.mark.differential


def _extract(
    recording: Recording,
    pages_dir: Path,
    output_dir: Path,
    config: ExtractionConfig | None = None,
) -> None:
    """Extract the recorded book again, into *output_dir*."""
    extractor = PageExtractor(
        config or replay_config(recording),
        detector=ReplayPredictor(recording),
        text_reader=ReplayTextExtractor(recording),
    )
    BookExtractor(extractor).extract(pages_dir, output_dir)


class TestRecordedBookReplays:
    def test_every_written_file_is_unchanged(
        self, recording: Recording, recorded_pages: Path, tmp_path: Path
    ) -> None:
        """The output tree matches the recording, path for path and byte for byte."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _extract(recording, recorded_pages, output_dir)

        assert directory_digests(output_dir) == recording.outputs

    def test_the_recording_is_not_empty(self, recording: Recording) -> None:
        """A recording of nothing would let the test above pass on nothing.

        Cheap, and it catches the one way this suite could go quietly useless:
        a recording made against a book that failed every page writes a file
        too, and every assertion over it would hold.
        """
        assert recording.outputs, f"{recording.source} recorded no output files"
        assert recording.detections, f"{recording.source} recorded no detections"


class TestTheComparisonCanFail:
    def test_a_different_size_cap_changes_the_images(
        self, recording: Recording, recorded_pages: Path, tmp_path: Path
    ) -> None:
        """A differential test that cannot tell two runs apart is worse than none.

        It reads as proof while asserting nothing. Capping questions at a
        different size is the smallest real change to the pipeline, and it has
        to show up in the digests.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        smaller = replace(
            replay_config(recording),
            question_max_width=max(1, recording.question_max_width // 2),
            question_max_height=max(1, recording.question_max_height // 2),
        )

        _extract(recording, recorded_pages, output_dir, smaller)

        assert directory_digests(output_dir) != recording.outputs
