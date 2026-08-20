"""Labelled training data, in nobody's format in particular.

The shape that travels from the annotation tool to the model trainer. It exists
because the two were once coupled directly: the YOLO dataset builder read Label
Studio's export JSON — its ``image`` key, its ``polygonlabels`` shape, its
percent coordinate space — while the layering contract claimed the trainer knew
nothing about Label Studio. The contract was describing a dependency that did
not hold.

By the time an annotation reaches here the vendor's half is done: the URI has
been resolved to a filename and the polygon converted out of the tool's percent
space into the normalized one label files use. What remains is what the trainer
actually needs — which is why it lives here rather than in either package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from digitex.domain.entities import NormalizedPolygon


@dataclass(frozen=True)
class LabelledRegion:
    """One polygon somebody drew, and what they called it.

    The label is the annotator's own word for the class. Turning it into a class
    id belongs to whoever writes a label file, because the id depends on which
    classes the export as a whole turned out to contain — a decision no single
    region can make.
    """

    label: str
    polygon: NormalizedPolygon


@dataclass(frozen=True)
class AnnotatedImage:
    """One image and every region drawn on it.

    ``filename`` is the image's name on disk rather than a path: the export
    names images by URI, and the pages themselves live wherever the caller
    keeps them. Two annotation batches can therefore each hold a ``30.jpg``,
    which is a collision for the consumer to notice — not something this shape
    can prevent.

    ``regions`` is empty for an image an annotator opened and drew nothing on.
    That is a legitimate negative example, not a malformed entry.
    """

    filename: str
    regions: tuple[LabelledRegion, ...] = ()
