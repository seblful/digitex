"""Handler for random question mode."""

import time

from aiogram import F, Router, types
from aiogram.fsm.context import FSMContext

from digitex.bot.database import with_uow
from digitex.bot.keyboards import part_a_kb, random_feedback_kb
from digitex.bot.messages import (
    MSG_CORRECT_ANSWER,
    MSG_ENTER_ANSWER,
    MSG_EXAM_CE,
    MSG_EXAM_CT,
    MSG_NO_RANDOM_QUESTION,
    MSG_NO_TOPIC_QUESTION,
    MSG_RANDOM_FINISH,
    MSG_WRONG_ANSWER,
)
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
    topic_name = data.get("topic_name")
    db_path = get_settings().database.path

    if topic_name:
        def fetch(uow):
            qid, part = uow.questions.get_random_question_id_by_topic(subject_id, topic_name)
            question = uow.questions.get(qid, part)
            origin = uow.questions.get_question_origin(qid)
            return question, origin

        try:
            question, origin = await with_uow(db_path, fetch)
        except KeyError:
            await message.answer(MSG_NO_TOPIC_QUESTION)
            return

        year, option, exam_type = origin

        await state.update_data(
            current_question_id=question.question_id,
            current_part=question.part,
            question_start_time=time.time(),
        )

        exam_type_label = MSG_EXAM_CE if exam_type == "CE" else MSG_EXAM_CT
        caption = (
            f"Тема: {topic_name}\n"
            f"<tg-spoiler>{exam_type_label} {year} год, вариант {option}</tg-spoiler>"
        )
    else:
        part = data["random_part"]
        exam_type = data.get("exam_type")

        def fetch(uow):
            qid = uow.questions.get_random_question_id(subject_id, part, exam_type)
            question = uow.questions.get(qid, part)
            origin = uow.questions.get_question_origin(qid)
            return question, origin

        try:
            question, origin = await with_uow(db_path, fetch)
        except KeyError:
            await message.answer(MSG_NO_RANDOM_QUESTION)
            return

        year, option, exam_type = origin

        await state.update_data(
            current_question_id=question.question_id,
            current_part=question.part,
            question_start_time=time.time(),
        )

        exam_type_label = MSG_EXAM_CE if exam_type == "CE" else MSG_EXAM_CT
        caption = f"<tg-spoiler>{exam_type_label} {year} год, вариант {option}</tg-spoiler>"

    if question.part == "A":
        await send_question(
            bot,
            message.chat.id,
            question,
            db_path,
            reply_markup=part_a_kb(question.num_options),
            caption=caption,
            parse_mode="HTML",
        )
    else:
        await send_question(
            bot, message.chat.id, question, db_path, caption=caption, parse_mode="HTML"
        )
        await message.answer(MSG_ENTER_ANSWER)

    await state.set_state(RandomTesting.answering)


@router.callback_query(RandomTesting.answering, F.data.startswith("ans:"))
async def on_random_part_a_answer(
    callback: types.CallbackQuery, state: FSMContext
) -> None:
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
    db_path = get_settings().database.path

    def check_answer(uow):
        correct = uow.questions.get_correct_answer(question_id, current_part)
        return correct

    correct_answer = await with_uow(db_path, check_answer)
    is_correct = answer.strip() == correct_answer.strip()

    if is_correct:
        await message.answer(MSG_CORRECT_ANSWER, reply_markup=random_feedback_kb())
    else:
        await message.answer(
            MSG_WRONG_ANSWER.format(correct_answer=correct_answer),
            reply_markup=random_feedback_kb(),
            parse_mode="HTML",
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
    await callback.message.answer(MSG_RANDOM_FINISH)
    await state.clear()
    await callback.answer()
