"""Which extractions have finished, kept in a JSON file between runs.

One concrete tracker and no abstract base. The extraction run is the only
caller, and pointing a tracker at a real file under ``tmp_path`` is a better
test stand-in than a subclass would be — an interface here is worth writing the
day a second store actually exists.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()


class JSONProgressTracker:
    """Remembers which ``(subject, identifier)`` extractions came through.

    The log is read once in ``__init__`` and written on every
    :meth:`mark_completed`, so a caller never has to remember a separate save
    and an interrupted run keeps whatever it had already finished. A file that
    is missing or unreadable starts an empty log rather than raising: the worst
    it costs is re-extracting a year, and refusing to start would cost the run.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._completed: dict[str, set[str]] = self._read()

    def is_completed(self, subject: str, identifier: str) -> bool:
        """Return True if this subject/identifier pair is already extracted."""
        return identifier in self._completed.get(subject, set())

    def mark_completed(self, subject: str, identifier: str) -> None:
        """Record the pair as extracted and write the log to disk."""
        self._completed.setdefault(subject, set()).add(identifier)
        self._save()

    def _read(self) -> dict[str, set[str]]:
        if not self._path.exists():
            return {}

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Failed to load progress file, starting fresh",
                path=str(self._path),
                error=str(e),
            )
            return {}

        logger.debug("Loaded progress", path=str(self._path), subjects=len(data))
        return {subject: set(done) for subject, done in data.items()}

    def _save(self) -> None:
        # Sorted, because a set's iteration order would rewrite the whole file
        # on every run and make the diff unreadable.
        data = {subject: sorted(done) for subject, done in self._completed.items()}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Saved progress", path=str(self._path))
