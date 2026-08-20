"""The layer everything else is allowed to depend on.

Pure Python and pydantic: the exam entities, the answer-matching rules, the
question-numbering state machine, and the on-disk corpus layout. Nothing here
imports a database driver, an image library, a model runtime or a web framework
— which is what makes it safe for both the bot and the studio to build on, and
what lets `digitex-core` be the one distribution both other members require.

Import the concrete module rather than this package (``from
digitex.domain.entities import Question``), the way the rest of the project does.
"""
