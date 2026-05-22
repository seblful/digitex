"""Repository classes — the only layer that touches raw SQL.

Each repository owns one aggregate (book, question, student, session,
authorized_user). Row containers and SQL fragments shared across them live
in :mod:`._common`.
"""

from digitex.core.db.repositories._common import (
    QuestionOrigin,
    SessionInfo,
    SubjectRow,
    WrongAnswer,
)
from digitex.core.db.repositories.authorized_user import AuthorizedUserRepository
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
    "authorized_users": AuthorizedUserRepository,
}


__all__ = [
    "REPOSITORIES",
    "AuthorizedUserRepository",
    "BookRepository",
    "QuestionOrigin",
    "QuestionRepository",
    "SessionInfo",
    "SessionRepository",
    "StudentRepository",
    "SubjectRow",
    "WrongAnswer",
]
