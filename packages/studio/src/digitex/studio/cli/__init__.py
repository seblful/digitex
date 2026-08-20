"""Entry points for the three studio workflows.

`digitex-extract` turns scans into question images, `digitex-train` fits the
segmentation model, `digitex-label` drives the annotation server. Each command
is an adapter: resolve `Settings`, build the objects, hand back values the
terminal can render.
"""
