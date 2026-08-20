"""The model: training it, and running it over a page.

Two subjects. :mod:`predictors` runs a trained checkpoint over one page image
and hands back labelled polygons; :mod:`yolo` builds the dataset and drives the
run that produced the checkpoint.

Nothing is re-exported here. Import the concrete module (``from
digitex.ml.predictors import YOLO_SegmentationPredictor``), the same rule
``pipeline`` and ``labeling`` follow — and here it also keeps ``import
digitex.ml`` from dragging in torch, which is most of what
:mod:`digitex.pipeline.ports` exists to avoid.

Off-limits to everything that deploys: the production image installs neither
torch nor ultralytics, so an import from ``bot``, ``db``, ``domain`` or
``config`` is an ImportError on the VPS. ``lint-imports`` is what catches it
here instead.
"""
