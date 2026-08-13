"""Repository classes — the only layer that touches raw SQL.

Each repository owns one aggregate (book, question, student, session). The
shapes they return live in :mod:`digitex.core.domain`, because callers outside
this layer read them.
"""

from digitex.core.db.repositories.book import BookRepository
from digitex.core.db.repositories.question import QuestionRepository
from digitex.core.db.repositories.session import SessionRepository
from digitex.core.db.repositories.student import StudentRepository

# Registry consumed by UnitOfWork to wire up all repositories on entry.
# Adding a new aggregate is "create the file, add one line here."
REPOSITORIES = {
    "books": BookRepository,
    "questions": QuestionRepository,
    "students": StudentRepository,
    "sessions": SessionRepository,
}


__all__ = [
    "REPOSITORIES",
    "BookRepository",
    "QuestionRepository",
    "SessionRepository",
    "StudentRepository",
]
