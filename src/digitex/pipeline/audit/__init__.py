"""Checks over the extraction output tree, carved out of CLI command bodies.

:mod:`census` counts what a subject produced and judges it complete or not;
:mod:`validator` checks each ``answers.json`` against the images beside it.
Each is a plain class that takes its inputs at construction and exposes a
single method, so the rules can be asserted on rather than inferred from a
terminal colour. Both are read by the review window as well as the CLI.
"""
