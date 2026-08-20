"""The bot's ports, and the Postgres classes that answer to them.

Two things are worth asserting about an interface introduced to invert a
dependency: that the implementation still satisfies it, and that the dependency
actually inverted. `lint-imports` states the second as a contract; this states
the first, so a method renamed on one side fails here rather than at runtime in
front of a student.

The check is method presence — `runtime_checkable` gives no more than that.
Whether the *signatures* line up is `ty`'s job, and it does it at every call
site: the first draft of these protocols guessed three parameter names wrong
and the type checker named all four call sites.
"""

from __future__ import annotations

import inspect
import re
import subprocess
import sys

import pytest

from digitex.domain.ports import (
    CatalogIndex,
    FileIdCache,
    QuestionCatalog,
    QuestionDraw,
    Repositories,
    SessionLog,
    StudentDirectory,
    TopicIndex,
)


@pytest.mark.parametrize(
    ("port", "adapter"),
    [
        (QuestionCatalog, "QuestionCatalog"),
        (QuestionDraw, "QuestionDraw"),
        (TopicIndex, "TopicIndex"),
        (FileIdCache, "FileIdCache"),
        (CatalogIndex, "BookRepository"),
        (SessionLog, "SessionRepository"),
        (StudentDirectory, "StudentRepository"),
    ],
)
def test_the_postgres_class_answers_to_its_port(port: type, adapter: str) -> None:
    repositories = pytest.importorskip("digitex.db.repositories")

    assert issubclass(getattr(repositories, adapter), port)


def test_every_role_the_bot_asks_for_is_one_a_transaction_provides() -> None:
    """`Repositories` names roles; the unit of work assigns them on entry.

    Checked against the names rather than with `isinstance`, for a reason worth
    keeping: the roles are set in `__aenter__`, so an unentered unit of work
    genuinely does not satisfy this protocol. What the protocol describes is
    what entering one hands back.

    Nothing in `digitex.db` imports the protocol or declares that it implements
    it, which is the point of structural typing here — the bot's requirements
    are stated in the bot's own layer.
    """
    unit_of_work = pytest.importorskip("digitex.db.unit_of_work")

    required = set(re.findall(r"def (\w+)\(self\)", inspect.getsource(Repositories)))
    provided = set(
        re.findall(
            r"self\.(\w+) = ", inspect.getsource(unit_of_work.UnitOfWork.__aenter__)
        )
    )

    assert required, "the protocol declares no roles — the regex stopped matching"
    missing = required - provided
    assert not missing, f"a transaction provides no {', '.join(sorted(missing))}"


def test_the_ports_need_nothing_at_runtime() -> None:
    """They are imported by the layer that must not grow a dependency.

    Every name they are built from sits behind `TYPE_CHECKING`, including the
    `OpenUow` alias — so importing them costs the deployed bot nothing beyond
    `typing`, and can never be the reason a production image needs a package.
    """
    code = (
        "import sys;"
        " import digitex.domain.ports;"
        " print('\\n'.join(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    loaded = set(result.stdout.split())

    assert loaded.isdisjoint({"psycopg", "psycopg_pool", "aiogram", "contextlib"})
