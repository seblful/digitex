"""Bot schemas — typed return values from repositories."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from digitex.core.schemas import Question, Session, Student, TestResult

__all__ = ["AuthorizedUser", "Question", "Session", "Student", "TestResult"]


class AuthorizedUser(BaseModel):
    """A user's registration/authorization record."""

    telegram_id: int
    full_name: str
    telegram_username: str | None = None
    status: Literal["pending", "approved", "rejected"]
    created_at: datetime
    handled_at: datetime | None = None
    handled_by: int | None = None
