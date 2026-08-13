"""The layer everything else is allowed to depend on.

Pure Python and pydantic: the exam entities, the answer-matching rules, and the
on-disk corpus layout. Nothing here imports a database driver, an image
library, a model runtime or a web framework — which is what makes it safe for
both the bot and the extraction pipeline to build on.

Import the concrete module rather than this package (``from
digitex.domain.entities import Question``), the same way the rest of the
project does.
"""
