"""Recorded detections and OCR answers, so an extraction run replays exactly.

Page extraction is deterministic given three answers it cannot work out for
itself: what the segmentation model found on a page, what OCR read off a
marker, and how far OCR thinks a piece is tilted. Everything after those —
masking, deskewing, stacking, capping, numbering — is arithmetic on pixels.

Recording those three and replaying them makes a run byte-reproducible on a
machine with no checkpoint, no GPU and no tesseract. Which is what lets a
restructuring prove it changed no output: extract a book once through each
implementation and compare the digests of the files written.

Answers are keyed by the digest of the image they were read off rather than by
the polygon that cut it, because ``detect_skew`` is handed a finished crop and
never sees a polygon. Keying by content also means a recording survives its
pages being renamed, and that two identical crops share one answer.

A recording is a test fixture, not corpus data. It is small — polygons and a
few hundred short strings — but the pages it refers to are not, so it stores
their names and leaves them in the book they came from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from digitex.domain.corpus import file_digest
from digitex.domain.entities import Detection, PixelPolygon
from digitex.pipeline.base import ExtractionConfig

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

    from digitex.pipeline.ports import RegionDetector, TextReader

RECORDING_VERSION = 1
"""Bumped when the on-disk shape changes, so a stale fixture fails loudly."""


class MissingAnswer(LookupError):
    """A replayed run asked something the recording does not hold.

    Always a real problem rather than a fixture to patch up: it means the run
    being replayed fed OCR an image the recorded run never produced, so the
    two runs have already diverged by the time this is raised.
    """

    def __init__(self, kind: str, digest: str) -> None:
        super().__init__(
            f"No recorded {kind} for image {digest[:12]} — the replayed run"
            " produced an image the recorded run did not."
        )
        self.kind = kind
        self.digest = digest


def image_digest(image: Image.Image) -> str:
    """Content digest of a PIL image, stable across saves and reloads.

    Taken off the raw pixel buffer together with the mode and size rather than
    off an encoded file, so it does not depend on a format's compression
    settings or metadata.
    """
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.size[0]}x{image.size[1]}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _detection_to_json(detection: Detection) -> dict[str, Any]:
    return {
        "label": detection.label,
        "polygon": [[x, y] for x, y in detection.polygon],
        "score": detection.score,
    }


def _detection_from_json(raw: dict[str, Any]) -> Detection:
    points = [(int(point[0]), int(point[1])) for point in raw["polygon"]]
    return Detection(
        label=str(raw["label"]),
        polygon=PixelPolygon(points),
        score=float(raw["score"]),
    )


@dataclass
class Recording:
    """Every answer one extraction run needed, plus what it wrote.

    ``pages`` is the page file names in the order the run consumed them, which
    is the order a replay has to feed them back in — the numbering state is
    threaded across pages, so a reordered replay is a different run.

    ``outputs`` maps each written file's path, relative to the year directory,
    to its content digest. That mapping is the actual assertion a differential
    run makes; everything else in here exists to make reproducing it possible.
    """

    source: str = ""
    """Which book this came off, as ``subject/year`` — for a human reading it."""

    image_format: str = "jpg"
    question_max_width: int = 0
    question_max_height: int = 0
    """The size cap and format the recorded run wrote with.

    Carried because they change the bytes of every file: a replay that used the
    settings live on the replaying machine would report a difference whenever
    those had been retuned since, which is not what this fixture measures.
    ``model_path`` is not carried — a replay never loads a checkpoint.
    """

    pages: list[str] = field(default_factory=list)
    detections: dict[str, list[Detection]] = field(default_factory=dict)
    text: dict[str, str] = field(default_factory=dict)
    digits: dict[str, list[int]] = field(default_factory=dict)
    skew: dict[str, float] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to indented JSON, with keys sorted so diffs stay readable."""
        payload = {
            "version": RECORDING_VERSION,
            "source": self.source,
            "image_format": self.image_format,
            "question_max_width": self.question_max_width,
            "question_max_height": self.question_max_height,
            "pages": self.pages,
            "detections": {
                digest: [_detection_to_json(d) for d in detections]
                for digest, detections in self.detections.items()
            },
            "text": self.text,
            "digits": self.digits,
            "skew": self.skew,
            "outputs": self.outputs,
        }
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)

    def write(self, path: Path) -> None:
        """Write the recording to *path*, creating its parent directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Recording:
        """Read a recording back.

        Raises:
            ValueError: If the file was written by a different format version.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        version = raw.get("version")
        if version != RECORDING_VERSION:
            raise ValueError(
                f"{path.name} is a version {version} recording, but this build"
                f" reads version {RECORDING_VERSION} — re-record it."
            )
        return cls(
            source=raw["source"],
            image_format=raw["image_format"],
            question_max_width=int(raw["question_max_width"]),
            question_max_height=int(raw["question_max_height"]),
            pages=list(raw["pages"]),
            detections={
                digest: [_detection_from_json(d) for d in detections]
                for digest, detections in raw["detections"].items()
            },
            text=dict(raw["text"]),
            digits={digest: list(values) for digest, values in raw["digits"].items()},
            skew={digest: float(value) for digest, value in raw["skew"].items()},
            outputs=dict(raw["outputs"]),
        )


def golden_dir(data_root: Path) -> Path:
    """Where recordings and their replay output live, under the data root.

    A fixture rather than corpus data, but it is still not code, so it hangs
    off ``data_root`` like everything else that is not.
    """
    return data_root / "golden"


def recording_path(data_root: Path, subject: str, year: str) -> Path:
    """The recording for one book."""
    return golden_dir(data_root) / f"{subject}-{year}.json"


def recorded_output_dir(data_root: Path, subject: str, year: str) -> Path:
    """Where a recording run writes its question images.

    Beside the recording rather than into the real extraction tree: a
    recording run starts from an empty folder so its numbering starts at 1,
    and it must never be able to overwrite the corpus the bot serves.
    """
    return golden_dir(data_root) / f"{subject}-{year}" / "output"


def directory_digests(root: Path) -> dict[str, str]:
    """Digest every file under *root*, keyed by its path relative to *root*.

    Paths use forward slashes whatever the platform, so a recording taken on
    Windows compares against a replay on Linux.
    """
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[path.relative_to(root).as_posix()] = file_digest(path)
    return digests


class RecordingPredictor:
    """Wraps the real predictor and keeps every answer it gives."""

    def __init__(self, predictor: RegionDetector, recording: Recording) -> None:
        self._predictor = predictor
        self._recording = recording

    def predict(self, image: Image.Image) -> list[Detection]:
        detections = self._predictor.predict(image)
        self._recording.detections[image_digest(image)] = detections
        return detections


class RecordingTextExtractor:
    """Wraps the real OCR and keeps every answer it gives."""

    def __init__(self, extractor: TextReader, recording: Recording) -> None:
        self._extractor = extractor
        self._recording = recording

    def extract_text(self, image: Image.Image, **kwargs: Any) -> str:
        text = self._extractor.extract_text(image, **kwargs)
        self._recording.text[image_digest(image)] = text
        return text

    def extract_digits(self, image: Image.Image, **kwargs: Any) -> list[int]:
        digits = self._extractor.extract_digits(image, **kwargs)
        self._recording.digits[image_digest(image)] = digits
        return digits

    def detect_skew(self, image: Image.Image, **kwargs: Any) -> float:
        angle = self._extractor.detect_skew(image, **kwargs)
        self._recording.skew[image_digest(image)] = angle
        return angle


class ReplayPredictor:
    """Hands back the recorded detections for a page, or refuses.

    Refusing rather than returning nothing is deliberate: an empty detection
    list is a legal answer that the extractor turns into "no detections found
    on page", which would read as a behaviour difference rather than as a
    fixture that does not cover the run.
    """

    def __init__(self, recording: Recording) -> None:
        self._recording = recording

    def predict(self, image: Image.Image) -> list[Detection]:
        digest = image_digest(image)
        try:
            return list(self._recording.detections[digest])
        except KeyError:
            raise MissingAnswer("detections", digest) from None


class ReplayTextExtractor:
    """Hands back recorded OCR answers, or refuses."""

    def __init__(self, recording: Recording) -> None:
        self._recording = recording

    def extract_text(self, image: Image.Image, **_kwargs: Any) -> str:
        digest = image_digest(image)
        try:
            return self._recording.text[digest]
        except KeyError:
            raise MissingAnswer("text", digest) from None

    def extract_digits(self, image: Image.Image, **_kwargs: Any) -> list[int]:
        digest = image_digest(image)
        try:
            return list(self._recording.digits[digest])
        except KeyError:
            raise MissingAnswer("digits", digest) from None

    def detect_skew(self, image: Image.Image, **_kwargs: Any) -> float:
        digest = image_digest(image)
        try:
            return self._recording.skew[digest]
        except KeyError:
            raise MissingAnswer("skew", digest) from None


def replay_config(recording: Recording) -> ExtractionConfig:
    """The extraction config the recorded run used.

    Since ``ExtractionConfig`` stopped carrying a model path this is only the
    size cap and the format — but those change the bytes of every file, so a
    replay that used the settings live on the replaying machine would report a
    difference whenever they had been retuned since.
    """
    return ExtractionConfig(
        image_format=recording.image_format,
        question_max_width=recording.question_max_width,
        question_max_height=recording.question_max_height,
    )
