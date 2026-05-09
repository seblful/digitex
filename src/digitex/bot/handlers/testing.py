"""Question answering loop — Part A (callbacks) and Part B (text)."""

import time

from aiogram import Bot, F, Router, types
from aiogram.fsm.context import FSMContext

from digitex.bot.database import with_uow
from digitex.bot.handlers.results import show_results
from digitex.bot.keyboards import part_a_kb
from digitex.bot.messages import MSG_ENTER_ANSWER
from digitex.bot.renderer import send_question
from digitex.bot.states import Testing
from digitex.core.answer import check_answer

router = Router()


async def send_current_question(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    db_path: str,
) -> None:
    data = await state.get_data()
    question_ids: list[tuple[int, str]] = data["question_ids"]
    current_index: int = data["current_index"]

    if current_index >= len(question_ids):
        await show_results(message, state, bot, db_path)
        return

    question_id, part = question_ids[current_index]

    def fetch_question(uow):
        return uow.questions.get(question_id, part)

    question = await with_uow(db_path, fetch_question)

    await state.update_data(
        question_start_time=time.time(),
        current_part=part,
        waiting_for_answer=True,
    )

    if part == "A":
        new_file_id = await send_question(
            bot,
            message.chat.id,
            question,
            reply_markup=part_a_kb(question.num_options),
        )
    else:
        new_file_id = await send_question(bot, message.chat.id, question)
        await message.answer(MSG_ENTER_ANSWER)

    if new_file_id:
        qid, qpart = question.question_id, question.part

        def cache(uow):
            uow.questions.cache_file_id(qid, qpart, new_file_id)

        await with_uow(db_path, cache)


async def _record_and_advance(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    answer: str,
    db_path: str,
) -> None:
    data = await state.get_data()
    session_id: int = data["session_id"]
    question_ids: list[tuple[int, str]] = data["question_ids"]
    current_index: int = data["current_index"]
    question_id, part = question_ids[current_index]
    question_start_time: float = data.get("question_start_time", time.time())

    time_spent = time.time() - question_start_time

    def record(uow):
        correct = uow.questions.get_correct_answer(question_id, part)
        is_correct = check_answer(part, answer, correct)
        uow.sessions.record_answer(
            session_id=session_id,
            question_id=question_id,
            student_answer=answer.strip(),
            is_correct=is_correct,
            time_spent=time_spent,
        )

    await with_uow(db_path, record)

    await state.update_data(current_index=current_index + 1)
    await send_current_question(message, state, bot, db_path)


@router.callback_query(Testing.answering, F.data.startswith("ans:"))
async def on_part_a_answer(
    callback: types.CallbackQuery, state: FSMContext, db_path: str
) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    answer = callback.data.split(":")[1]
    await _record_and_advance(callback.message, state, callback.bot, answer, db_path)
    await callback.answer()


@router.message(Testing.answering)
async def on_part_b_answer(
    message: types.Message, state: FSMContext, db_path: str
) -> None:
    if not message.text:
        return

    data = await state.get_data()
    current_part = data.get("current_part")
    waiting = data.get("waiting_for_answer", False)

    if current_part != "B" or not waiting:
        return

    await state.update_data(waiting_for_answer=False)
    await _record_and_advance(message, state, message.bot, message.text, db_path)
