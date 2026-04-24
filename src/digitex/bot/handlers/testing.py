"""Question answering loop — Part A (callbacks) and Part B (text)."""

import time

from aiogram import F, Router, types
from aiogram.fsm.context import FSMContext

from digitex.bot.database import with_uow
from digitex.bot.keyboards import part_a_kb
from digitex.bot.renderer import send_question
from digitex.bot.states import Testing
from digitex.config import get_settings

router = Router()


async def send_current_question(
    message: types.Message,
    state: FSMContext,
    bot,
) -> None:
    data = await state.get_data()
    question_ids: list[int] = data["question_ids"]
    current_index: int = data["current_index"]

    if current_index >= len(question_ids):
        from digitex.bot.handlers.results import show_results
        await show_results(message, state, bot)
        return

    question_id = question_ids[current_index]
    db_path = get_settings().database.path

    def fetch_question(uow):
        question = uow.questions.get(question_id)
        return question, question.part

    question, part = await with_uow(db_path, fetch_question)

    # Store timestamp when question is shown
    await state.update_data(question_start_time=time.time())

    if part == "A":
        await send_question(bot, message.chat.id, question, db_path, reply_markup=part_a_kb())
    else:
        await send_question(bot, message.chat.id, question, db_path)
        await message.answer("Введите ответ текстом:")


async def _record_and_advance(
    message: types.Message,
    state: FSMContext,
    bot,
    answer: str,
) -> None:
    data = await state.get_data()
    session_id: int = data["session_id"]
    question_ids: list[int] = data["question_ids"]
    current_index: int = data["current_index"]
    question_id = question_ids[current_index]
    question_start_time: float = data.get("question_start_time", time.time())
    db_path = get_settings().database.path

    time_spent = time.time() - question_start_time

    def record(uow):
        question = uow.questions.get(question_id)
        correct = uow.questions.get_correct_answer(question_id, question.part)
        is_correct = answer.strip() == correct.strip()
        uow.sessions.record_answer(
            session_id=session_id,
            question_id=question_id,
            student_answer=answer.strip(),
            is_correct=is_correct,
            time_spent=time_spent,
        )
        return is_correct

    await with_uow(db_path, record)

    await state.update_data(current_index=current_index + 1)
    await send_current_question(message, state, bot)


@router.callback_query(Testing.answering, F.data.startswith("ans:"))
async def on_part_a_answer(callback: types.CallbackQuery, state: FSMContext) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    answer = callback.data.split(":")[1]
    await _record_and_advance(callback.message, state, callback.bot, answer)
    await callback.answer()


@router.message(Testing.answering)
async def on_part_b_answer(message: types.Message, state: FSMContext) -> None:
    if not message.text:
        return
    await _record_and_advance(message, state, message.bot, message.text)
