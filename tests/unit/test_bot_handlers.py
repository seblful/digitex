"""Tests for the bot's result formatting and question rendering."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from digitex.bot.handlers.results import _format_result_lines
from digitex.bot.renderer import send_question
from digitex.core.db.repositories import SessionInfo, WrongAnswer
from digitex.core.domain import ExamType, Question, TestResult


class TestFormatResultLines:
    def _make_result(self, exam_type: ExamType = "CT") -> TestResult:
        return TestResult(
            session_id=1,
            exam_type=exam_type,
            part_a_score=8,
            part_b_score=6,
            total_score=14,
            max_score=20,
            time_spent=600.0,
            completed_at=datetime.now(UTC),
        )

    def _make_info(self) -> SessionInfo:
        return SessionInfo(subject_name="Физика", year=2023, option_number=3)

    def test_no_wrong_answers(self) -> None:
        lines = _format_result_lines(self._make_result(), [], self._make_info())
        text = "\n".join(lines)
        assert "Физика" in text
        assert "2023" in text
        assert "3" in text
        assert "14" in text

    def test_wrong_answers_appear(self) -> None:
        wrong = [
            WrongAnswer(
                question_number=2, part="A", student_answer="3", correct_answer="4"
            ),
            WrongAnswer(
                question_number=1, part="B", student_answer="xyz", correct_answer="abc"
            ),
        ]
        lines = _format_result_lines(self._make_result(), wrong, self._make_info())
        text = "\n".join(lines)
        assert "xyz" in text
        assert "abc" in text

    def test_exam_type_ce_label(self) -> None:
        lines = _format_result_lines(
            self._make_result(exam_type="CE"), [], self._make_info()
        )
        text = "\n".join(lines)
        assert "ЦЭ" in text or "CE" in text

    def test_exam_type_ct_label(self) -> None:
        lines = _format_result_lines(
            self._make_result(exam_type="CT"), [], self._make_info()
        )
        text = "\n".join(lines)
        assert "ЦТ" in text or "CT" in text


class TestSendQuestion:
    async def test_sends_with_file_id_when_cached(self) -> None:
        bot = AsyncMock()
        question = Question(
            question_id=1,
            part="A",
            question_number=1,
            image_data=b"fake",
            telegram_file_id="cached_file_id",
        )
        result = await send_question(bot, 1, question)
        bot.send_photo.assert_awaited_once()
        call_kwargs = bot.send_photo.call_args.kwargs
        assert call_kwargs["photo"] == "cached_file_id"
        assert result is None

    async def test_uploads_and_returns_file_id_when_not_cached(self) -> None:
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

        result = await send_question(bot, 1, question)
        assert bot.send_photo.await_count == 1
        assert result == "new_file_id"

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

        with patch("digitex.bot.renderer.logger") as mock_logger:
            result = await send_question(bot, 1, question)
        mock_logger.warning.assert_called_once()
        assert result is None
