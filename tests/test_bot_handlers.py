"""Tests for bot handlers."""

import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.types import Chat, Message, User

from digitex.bot.database import with_uow
from digitex.bot.renderer import send_question
from digitex.bot.schemas import Question


def _make_message(
    chat_id: int = 1,
    user_id: int = 123,
    full_name: str = "Test User",
    username: str | None = "@testuser",
) -> MagicMock:
    chat = MagicMock(spec=Chat)
    chat.id = chat_id
    user = MagicMock(spec=User)
    user.id = user_id
    user.full_name = full_name
    user.username = username
    msg = MagicMock(spec=Message)
    msg.chat = chat
    msg.from_user = user
    msg.answer = AsyncMock()
    return msg


class TestWithUow:
    def test_runs_callback_in_executor(self) -> None:
        db_path = ":memory:"

        def init_db():
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
            conn.commit()
            conn.close()

        init_db()

        async def run_test() -> int:
            def callback(uow):
                return 42

            return await with_uow(db_path, callback)

        import asyncio
        result = asyncio.run(run_test())
        assert result == 42


class TestSendQuestion:
    @pytest.mark.asyncio
    async def test_sends_with_file_id_when_cached(self) -> None:
        bot = AsyncMock()
        question = Question(
            question_id=1,
            part="A",
            question_number=1,
            image_data=b"fake",
            telegram_file_id="cached_file_id",
        )
        await send_question(bot, 1, question, "fake.db")
        bot.send_photo.assert_called_once()
        call_kwargs = bot.send_photo.call_args.kwargs
        assert call_kwargs["photo"] == "cached_file_id"

    @pytest.mark.asyncio
    async def test_uploads_and_caches_when_no_file_id(self) -> None:
        bot = AsyncMock()
        photo_msg = MagicMock()
        photo_msg.photo = [MagicMock(file_id="new_file_id")]
        bot.send_photo.return_value = photo_msg

        question = Question(
            question_id=1,
            part="A",
            question_number=1,
            image_data=b"fake",
            telegram_file_id=None,
        )

        with patch("digitex.bot.renderer.with_uow", new_callable=AsyncMock) as mock_uow:
            await send_question(bot, 1, question, "fake.db")
            assert bot.send_photo.call_count == 1
            mock_uow.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_warning_when_no_photo_in_response(self) -> None:
        bot = AsyncMock()
        photo_msg = MagicMock()
        photo_msg.photo = []
        bot.send_photo.return_value = photo_msg

        question = Question(
            question_id=1,
            part="A",
            question_number=1,
            image_data=b"fake",
            telegram_file_id=None,
        )

        with patch("digitex.bot.renderer.with_uow", new_callable=AsyncMock):
            with patch("digitex.bot.renderer.logger") as mock_logger:
                await send_question(bot, 1, question, "fake.db")
                mock_logger.warning.assert_called_once()
