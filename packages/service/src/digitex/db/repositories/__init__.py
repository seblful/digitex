"""Repository classes — the only layer that touches raw SQL.

Each class owns one role: a book, a student, a session, and the five a
question is addressed through — reading one to serve, drawing one at random,
the topic map, the Telegram file_id cache, and loading the corpus in. The
shapes they return live in :mod:`digitex.domain.entities`, because callers
outside this layer read them.
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
