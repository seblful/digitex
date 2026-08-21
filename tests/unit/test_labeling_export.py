"""Tests for reading a Label Studio export into annotations.

Every assumption about the tool's JSON is in the module under test, so this is
where a malformed export is exercised: a region with no label, a region with no
points, an entry whose URI names no local file. A partially broken export has
to yield a usable dataset, because the alternative is one bad polygon in a
batch of six hundred images costing a training run.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from structlog.testing import capture_logs

from digitex.labeling.export import read_export

if TYPE_CHECKING:
    from pathlib import Path

_SQUARE = [[10.0, 20.0], [50.0, 20.0], [50.0, 80.0], [10.0, 80.0]]


def _write(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _entry(uri: str, labels: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "image": uri,
        "label": [{"polygonlabels": ["question"], "points": _SQUARE}]
        if labels is None
        else labels,
    }


class TestReadExport:
    def test_a_percent_polygon_comes_back_normalized(self, tmp_path: Path) -> None:
        """The one hop out of the tool's coordinate space."""
        path = _write(tmp_path, [_entry("/data/local-files/?d=page.jpg")])

        images = read_export(path)

        assert len(images) == 1
        assert images[0].filename == "page.jpg"
        assert images[0].regions[0].label == "question"
        assert list(images[0].regions[0].polygon) == [
            (0.1, 0.2),
            (0.5, 0.2),
            (0.5, 0.8),
            (0.1, 0.8),
        ]

    def test_a_url_encoded_uri_decodes_to_a_filename(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [_entry("/data/local-files/?d=images%5Cmy%20file.jpg")])

        images = read_export(path)

        assert images[0].filename == "my file.jpg"

    def test_an_entry_naming_no_local_file_is_dropped(self, tmp_path: Path) -> None:
        """A task synced from blob storage has no page on this disk to train on."""
        path = _write(tmp_path, [_entry("https://example.test/page.jpg")])

        with capture_logs() as logs:
            images = read_export(path)

        assert images == []
        assert any(entry["event"] == "skipped_entry_no_local_path" for entry in logs)

    def test_a_region_missing_its_label_or_points_is_skipped(
        self, tmp_path: Path
    ) -> None:
        path = _write(
            tmp_path,
            [
                _entry(
                    "/data/local-files/?d=page.jpg",
                    labels=[
                        {"polygonlabels": [], "points": [[10.0, 20.0]]},
                        {"polygonlabels": ["question"], "points": []},
                    ],
                )
            ],
        )

        with capture_logs() as logs:
            images = read_export(path)

        # The image survives with no regions rather than the read failing.
        assert images[0].regions == ()
        assert sum(entry["event"] == "skipped_polygon" for entry in logs) == 2

    def test_the_usable_regions_survive_beside_a_broken_one(
        self, tmp_path: Path
    ) -> None:
        path = _write(
            tmp_path,
            [
                _entry(
                    "/data/local-files/?d=page.jpg",
                    labels=[
                        {"polygonlabels": ["question"], "points": []},
                        {"polygonlabels": ["question"], "points": _SQUARE},
                    ],
                )
            ],
        )

        images = read_export(path)

        assert len(images[0].regions) == 1

    def test_a_duplicate_basename_is_returned_not_resolved(
        self, tmp_path: Path
    ) -> None:
        """Two batches can each hold a 30.jpg; deciding what to do is not this.

        The export addresses images by URI and only the basename survives, so
        the collision is real — but whoever assembles a dataset out of these is
        the one that has to notice it.
        """
        path = _write(
            tmp_path,
            [
                _entry("/data/local-files/?d=batch1%5C30.jpg"),
                _entry("/data/local-files/?d=batch2%5C30.jpg"),
            ],
        )

        images = read_export(path)

        assert [image.filename for image in images] == ["30.jpg", "30.jpg"]

    def test_an_empty_export_reads_as_nothing(self, tmp_path: Path) -> None:
        assert read_export(_write(tmp_path, [])) == []

    def test_a_closed_ring_loses_its_closing_point(self, tmp_path: Path) -> None:
        """A kept pre-annotation that arrived closed, on its way to a label file.

        Label Studio stores an open ring, so the repeat is not the tool's: 116
        of the training set's polygons carry one from predictions that were
        uploaded closed. The vertex says nothing about the region either way.
        """
        closed = [*_SQUARE, _SQUARE[0]]
        path = _write(
            tmp_path,
            [
                _entry(
                    "/data/local-files/?d=page.jpg",
                    [{"polygonlabels": ["question"], "points": closed}],
                )
            ],
        )

        images = read_export(path)

        polygon = list(images[0].regions[0].polygon)
        assert polygon == [(0.1, 0.2), (0.5, 0.2), (0.5, 0.8), (0.1, 0.8)]

    def test_a_polygon_that_merely_returns_near_its_start_is_left_alone(
        self, tmp_path: Path
    ) -> None:
        """Only an exact repeat is a closing point; a near miss is a vertex."""
        nearly = [*_SQUARE, [10.1, 20.0]]
        path = _write(
            tmp_path,
            [
                _entry(
                    "/data/local-files/?d=page.jpg",
                    [{"polygonlabels": ["question"], "points": nearly}],
                )
            ],
        )

        images = read_export(path)

        assert len(images[0].regions[0].polygon) == 5
