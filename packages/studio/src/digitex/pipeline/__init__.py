"""Extraction — question images out of scanned book pages.

Three nested runners, each doing one unit of the corpus and delegating the one
below it: :mod:`subject` walks a subject's years, :mod:`book` walks a year's
pages, :mod:`page` turns one page into question crops. :mod:`answers` is the
separate pass that reads a year's answer key off the back of the book, and
:mod:`audit` checks what came out.

Import the concrete module, not this package: ``from digitex.pipeline.book
import BookExtractor``. There is deliberately no re-export list here, so there
is only ever one spelling for a name.
"""
