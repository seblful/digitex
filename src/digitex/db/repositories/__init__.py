"""Repository classes — the only layer that touches raw SQL.

Each repository owns one aggregate (book, question, student, session). The
shapes they return live in :mod:`digitex.domain.entities`, because callers outside
this layer read them.
"""

from digitex.db.repositories.book import BookRepository
from digitex.db.repositories.question import QuestionRepository
from digitex.db.repositories.session import SessionRepository
from digitex.db.repositories.student import StudentRepository

__all__ = [
    "BookRepository",
    "QuestionRepository",
    "SessionRepository",
    "StudentRepository",
]
