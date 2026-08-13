"""Tests for the bot's FSM state groups.

aiogram routes an update by the *name* a state resolves to, so the one thing
worth pinning is that no two states share one. ``Testing`` and ``RandomTesting``
both declare ``answering``: were the group prefix ever dropped, a random-mode
reply would match the standard-mode handler and be scored against a question
from a different round.

That the declarations are ``State`` objects is guaranteed by ``StatesGroup``
itself, so there is nothing to check there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from digitex.bot.states import Navigation, RandomTesting, Registration, Testing

if TYPE_CHECKING:
    from aiogram.fsm.state import State, StatesGroup

GROUPS: tuple[type[StatesGroup], ...] = (
    Registration,
    Navigation,
    Testing,
    RandomTesting,
)


def _states() -> list[State]:
    return [state for group in GROUPS for state in group.__all_states__]


class TestStateNames:
    def test_the_sweep_found_every_group(self) -> None:
        """Guard the guard: a group left out of GROUPS would not be checked."""
        assert len(_states()) == 12

    def test_no_two_states_resolve_to_the_same_name(self) -> None:
        names = [state.state for state in _states()]

        assert len(set(names)) == len(names), f"duplicate state name in {names}"

    def test_the_two_answering_states_stay_apart(self) -> None:
        """The collision the group prefix exists to prevent, pinned by name."""
        assert Testing.answering.state != RandomTesting.answering.state
