"""The laptop half: scans in, a question corpus out.

Nothing here ships to the VPS. This package holds the entry points that
assemble the studio workflows — extraction, training, annotation — from the
layers below (`imaging`, `ml`, `labeling`, `pipeline`, `ui`), and sits above all
of them in the layering contract.
"""
