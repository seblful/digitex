"""Handler for random question mode."""

import time
from aiogram import F, Router, types
from aiogram.fsm.context import FSMContext

from digitex.bot.database import with_uow
from digitex.bot.keyboards import part_a_kb, random_feedback_kb
from digitex.bot.renderer import send_question
from digitex.bot.states import RandomTesting
from digitex.config import get_settings

router = Router()

async def start_random_question(
    message: types.Message,
    state: FSMContext,
    bot,
) -> None:
    data = await state.get_data()
    subject_id = data["subject_id"]
    part = data["random_part"]
    db_path = get_settings().database.path

    def fetch_random(uow):
        qid = uow.questions.get_random_question_id(subject_id, part)
        question = uow.questions.get(qid, part)
        origin = uow._conn.execute(
            "SELECT b.year_value, o.option_number"
            " FROM part_a_questions q"
            " JOIN options o ON q.option_id = o.option_id"
            " JOIN books b ON o.book_id = b.book_id"
            " WHERE q.question_id = ?"
            " UNION ALL"
            " SELECT b.year_value, o.option_number"
            " FROM part_b_questions q"
            " JOIN options o ON q.option_id = o.option_id"
            " JOIN books b ON o.book_id = b.book_id"
            " WHERE q.question_id = ?",
            (qid, qid),
        ).fetchone()
        return question, origin

    try:
        question, origin = await with_uow(db_path, fetch_random)
    except KeyError:
        await message.answer("Не удалось найти случайный вопрос.")
        return

    year, option = origin

    await state.update_data(
        current_question_id=question.question_id,
        current_part=question.part,
        question_start_time=time.time(),
        question_year=year,
        question_option=option,
    )
    
    if question.part == "A":
        await send_question(bot, message.chat.id, question, db_path, reply_markup=part_a_kb(question.num_options))
    else:
        await send_question(bot, message.chat.id, question, db_path)
        await message.answer("Введите ответ текстом:")
    
    await state.set_state(RandomTesting.answering)

@router.callback_query(RandomTesting.answering, F.data.startswith("ans:"))
async def on_random_part_a_answer(callback: types.CallbackQuery, state: FSMContext) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return
    
    answer = callback.data.split(":")[1]
    await process_random_answer(callback.message, state, callback.bot, answer)
    await callback.answer()

@router.message(RandomTesting.answering)
async def on_random_part_b_answer(message: types.Message, state: FSMContext) -> None:
    if not message.text:
        return
    
    data = await state.get_data()
    if data.get("current_part") != "B":
        return
        
    await process_random_answer(message, state, message.bot, message.text)

async def process_random_answer(
    message: types.Message,
    state: FSMContext,
    bot,
    answer: str,
) -> None:
    data = await state.get_data()
    question_id = data["current_question_id"]
    current_part = data["current_part"]
    year = data["question_year"]
    option = data["question_option"]
    db_path = get_settings().database.path

    def check_answer(uow):
        correct = uow.questions.get_correct_answer(question_id, current_part)
        return correct

    correct_answer = await with_uow(db_path, check_answer)
    is_correct = answer.strip() == correct_answer.strip()
    
    result = "✅" if is_correct else "❌"
    feedback = f"{result} {year} год, вариант {option} — {'Правильно!' if is_correct else f'Неправильно. Правильный ответ: {correct_answer}'}"
    
    await message.answer(
        feedback,
        reply_markup=random_feedback_kb()
    )
    await state.set_state(RandomTesting.feedback)

@router.callback_query(RandomTesting.feedback, F.data == "random:next")
async def on_random_next(callback: types.CallbackQuery, state: FSMContext) -> None:
    if not callback.message or not isinstance(callback.message, types.Message):
        await callback.answer("Ошибка: сообщение недоступно")
        return
    await start_random_question(callback.message, state, callback.bot)
    await callback.answer()

@router.callback_query(RandomTesting.feedback, F.data == "random:finish")
async def on_random_finish(callback: types.CallbackQuery, state: FSMContext) -> None:
    if not callback.message or not isinstance(callback.message, types.Message):
        await callback.answer("Ошибка: сообщение недоступно")
        return
    await callback.message.answer("Режим случайных вопросов завершен. Используйте /start для начала заново.")
    await state.clear()
    await callback.answer()
