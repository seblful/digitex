"""Label Studio — the only layer that knows the annotation tool exists.

:mod:`client` is the narrow adapter over its SDK, :mod:`uris` reads the URIs it
serves local images through, and :mod:`export` turns its export JSON into the
vendor-neutral shapes in :mod:`digitex.domain.annotations`. On top of those sit
the three jobs: :mod:`predictor` pre-annotates a project, :mod:`repair` fixes
one whose images moved out from under it, and :mod:`skipped` retires the pages
an annotator refused.

Import the concrete module, not this package: ``from digitex.labeling.client
import LabelStudioClient``. There is deliberately no re-export list here, so
there is only ever one spelling for a name — the same rule ``pipeline`` and
``ui`` follow.
"""
