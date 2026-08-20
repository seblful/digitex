"""Desktop windows for the steps that need a pair of eyes.

Import the concrete module rather than this package: ``from digitex.ui.page_review
import TkPageReviewer``. Tkinter is imported nowhere outside these modules, and
inside them only by the three that are windows, so `digitex.pipeline` and the CLI
stay importable on a machine with no display.
"""
