"""Labelled training data, in nobody's format in particular.

The shape that travels between the annotation tool and the model trainer. It
exists because the two were coupled directly: the YOLO dataset builder read
Label Studio's export JSON — its ``image`` key, its ``polygonlabels`` shape,
its percent coordinate space — while the layer contract said the trainer knows
nothing about Label Studio. The contract was describing a dependency that did
not hold.

An annotation reaching here has already been through the vendor's half: the
URI resolved to a filename, the polygon converted out of the tool's percent
space into the normalized one label files are written in. What is left is what
the trainer actually needs, which is why it is here and not in either package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from digitex.domain.entities import NormalizedPolygon


@dataclass(frozen=True)
class LabelledRegion:
    """One polygon somebody drew, and what they called it.

    The label is the annotator's own word for the class — mapping it to a class
    id belongs to whoever writes a label file, because the id depends on which
    classes the whole export turned out to contain.
    """

    label: str
    polygon: NormalizedPolygon


@dataclass(frozen=True)
class AnnotatedImage:
    """One image and every region drawn on it.

    ``filename`` is the image's name on disk, not a path: the export names
    images by URI and the pages themselves live wherever the caller keeps them.
    Two annotation batches can therefore both hold a ``30.jpg``, which is a
    collision the consumer has to notice rather than a shape this can prevent.

    ``regions`` is empty for an image an annotator opened and drew nothing on.
    That is a legitimate negative example, not a malformed entry.
    """

    filename: str
    regions: tuple[LabelledRegion, ...] = ()
