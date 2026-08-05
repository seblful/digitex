"""Per-year extraction progress, persisted as JSON.

One concrete tracker and no abstract base: the extraction run is the only
caller, and pointing a tracker at a real file under ``tmp_path`` is a better
test stand-in than a subclass. Introduce an interface here the day a second
store actually exists.
"""

import json
from pathlib import Path

import structlog

logger = structlog.get_logger()


class JSONProgressTracker:
    """Records which ``(subject, identifier)`` extractions have completed.

    ``mark_completed`` persists immediately, so callers never have to remember
    a separate save. Loading is done in ``__init__``; a missing or corrupt file
    starts an empty log rather than raising.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._completed: dict[str, set[str]] = {}
        self._load()

    def is_completed(self, subject: str, identifier: str) -> bool:
        """Return True if this subject/identifier pair is already extracted."""
        return identifier in self._completed.get(subject, set())

    def mark_completed(self, subject: str, identifier: str) -> None:
        """Record the pair as extracted and write the log to disk."""
        self._completed.setdefault(subject, set()).add(identifier)
        self._save()

    def _load(self) -> None:
        if not self._path.exists():
            self._completed = {}
            return

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._completed = {k: set(v) for k, v in data.items()}
            logger.debug("Loaded progress", path=str(self._path), subjects=len(data))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Failed to load progress file, starting fresh",
                path=str(self._path),
                error=str(e),
            )
            self._completed = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: sorted(v) for k, v in self._completed.items()}
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Saved progress", path=str(self._path))
