"""The only layer that writes SQL, split by role rather than by table.

Eight classes over eleven tables: a book, a student, a session, and the five
ways a question is addressed — reading one to serve, drawing one at random, the
topic map, Telegram's ``file_id`` cache, and loading the corpus in. They are
grouped by what a caller wants to do, because no caller wanted more than three
of the question roles and a class per aggregate made every one of them carry all
five.

The shapes they hand back live in :mod:`digitex.domain.entities`, since callers
outside this layer read them. What those callers are written against — the
protocols in ``digitex.domain.ports`` — is deliberately not imported here: the
fit is structural, and ``ty`` is what checks it.
"""

from digitex.db.repositories.book import BookRepository
from digitex.db.repositories.question import (
    FileIdCache,
    QuestionCatalog,
    QuestionCorpus,
    QuestionDraw,
    TopicIndex,
)
from digitex.db.repositories.session import SessionRepository
from digitex.db.repositories.student import StudentRepository

__all__ = [
    "BookRepository",
    "FileIdCache",
    "QuestionCatalog",
    "QuestionCorpus",
    "QuestionDraw",
    "SessionRepository",
    "StudentRepository",
    "TopicIndex",
]
