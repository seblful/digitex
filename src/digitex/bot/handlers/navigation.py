"""Subject → Year → Option selection callbacks."""

from aiogram import F, Router, types
from aiogram.fsm.context import FSMContext

from digitex.bot.constants import FALLBACK_NAME
from digitex.bot.database import with_uow
from digitex.bot.handlers.random import start_random_question
from digitex.bot.handlers.testing import send_current_question
from digitex.bot.keyboards import (
    exam_type_kb,
    mode_kb,
    options_kb,
    random_part_kb,
    subjects_kb,
    topics_kb,
    years_kb,
)
from digitex.bot.messages import (
    MSG_EXAM_TYPE_SELECT,
    MSG_MODE_SELECT,
    MSG_NO_OPTIONS,
    MSG_NO_TOPICS,
    MSG_NO_YEARS,
    MSG_OPTION_SELECT,
    MSG_PART_SELECT,
    MSG_START_TESTING,
    MSG_TOPIC_SELECT,
    MSG_YEAR_SELECT,
)
from digitex.bot.states import Navigation, Testing

router = Router()


@router.callback_query(Navigation.select_subject, F.data.startswith("subj:"))
async def on_subject_selected(callback: types.CallbackQuery, state: FSMContext) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    subject_id = int(callback.data.split(":")[1])
    data = await state.get_data()
    student_id = data.get("student_id")
    await state.clear()
    await state.update_data(subject_id=subject_id, student_id=student_id)
    await callback.message.edit_text(MSG_MODE_SELECT, reply_markup=mode_kb())
    await state.set_state(Navigation.select_mode)
    await callback.answer()


@router.callback_query(Navigation.select_mode, F.data.startswith("mode:"))
async def on_mode_selected(
    callback: types.CallbackQuery, state: FSMContext, db_path: str
) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    mode = callback.data.split(":")[1]
    data = await state.get_data()
    subject_id = data["subject_id"]

    match mode:
        case "standard":
            years = await with_uow(
                db_path, lambda uow: uow.books.list_years(subject_id)
            )
            if not years:
                subjects = await with_uow(
                    db_path, lambda uow: uow.books.list_subjects()
                )
                await callback.message.edit_text(
                    MSG_NO_YEARS,
                    reply_markup=subjects_kb(subjects),
                )
                await state.set_state(Navigation.select_subject)
                await callback.answer()
                return
            await callback.message.edit_text(
                MSG_YEAR_SELECT, reply_markup=years_kb(years)
            )
            await state.set_state(Navigation.select_year)

        case "random":
            await callback.message.edit_text(
                MSG_EXAM_TYPE_SELECT, reply_markup=exam_type_kb()
            )
            await state.set_state(Navigation.select_random_exam_type)

        case "topics":
            topics = await with_uow(
                db_path, lambda uow: uow.questions.get_topics_for_subject(subject_id)
            )
            if not topics:
                await callback.message.edit_text(MSG_NO_TOPICS)
                await callback.answer()
                return
            await callback.message.edit_text(
                MSG_TOPIC_SELECT, reply_markup=topics_kb(topics)
            )
            await state.update_data(topic_names=topics)
            await state.set_state(Navigation.select_topic)

    await callback.answer()


@router.callback_query(Navigation.select_topic, F.data.startswith("topic:"))
async def on_topic_selected(
    callback: types.CallbackQuery, state: FSMContext, db_path: str
) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    idx = int(callback.data.split(":")[1])
    data = await state.get_data()
    topic_name = data["topic_names"][idx]
    await state.update_data(topic_name=topic_name)

    await start_random_question(callback.message, state, callback.bot, db_path)
    await callback.answer()


@router.callback_query(
    Navigation.select_random_exam_type, F.data.startswith("exam_type:")
)
async def on_random_exam_type_selected(
    callback: types.CallbackQuery, state: FSMContext
) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    exam_type = callback.data.split(":")[1]
    await state.update_data(exam_type=exam_type)
    await callback.message.edit_text(MSG_PART_SELECT, reply_markup=random_part_kb())
    await state.set_state(Navigation.select_random_part)
    await callback.answer()


@router.callback_query(Navigation.select_random_part, F.data.startswith("random_part:"))
async def on_random_part_selected(
    callback: types.CallbackQuery, state: FSMContext, db_path: str
) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    part = callback.data.split(":")[1]
    await state.update_data(random_part=part)

    await start_random_question(callback.message, state, callback.bot, db_path)
    await callback.answer()


@router.callback_query(Navigation.select_year, F.data.startswith("year:"))
async def on_year_selected(
    callback: types.CallbackQuery, state: FSMContext, db_path: str
) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    year = int(callback.data.split(":")[1])
    await state.update_data(year=year)

    if year >= 2023:
        await callback.message.edit_text(
            MSG_EXAM_TYPE_SELECT,
            reply_markup=exam_type_kb(),
        )
        await state.set_state(Navigation.select_exam_type)
    else:
        await _show_options_for_exam_type(callback.message, state, year, "CT", db_path)

    await callback.answer()


@router.callback_query(Navigation.select_exam_type, F.data.startswith("exam_type:"))
async def on_exam_type_selected(
    callback: types.CallbackQuery, state: FSMContext, db_path: str
) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    exam_type = callback.data.split(":")[1]
    data = await state.get_data()
    year = data["year"]
    await _show_options_for_exam_type(callback.message, state, year, exam_type, db_path)
    await callback.answer()


async def _show_options_for_exam_type(
    message: types.Message,
    state: FSMContext,
    year: int,
    exam_type: str,
    db_path: str,
) -> None:
    data = await state.get_data()
    subject_id = data["subject_id"]

    def fetch_options(uow):
        book_id = uow.books.get_or_create_book(subject_id, year)
        options = uow.books.list_options(book_id, exam_type)
        return book_id, options

    book_id, options = await with_uow(db_path, fetch_options)
    if not options:
        await message.edit_text(MSG_NO_OPTIONS.format(exam_type=exam_type))
        await state.set_state(Navigation.select_year)
        return

    await message.edit_text(MSG_OPTION_SELECT, reply_markup=options_kb(options))
    await state.update_data(book_id=book_id, exam_type=exam_type)
    await state.set_state(Navigation.select_option)


@router.callback_query(Navigation.select_option, F.data.startswith("opt:"))
async def on_option_selected(
    callback: types.CallbackQuery, state: FSMContext, db_path: str
) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    option_number = int(callback.data.split(":")[1])
    data = await state.get_data()
    book_id = data["book_id"]

    def start_test(uow):
        student_id = data.get("student_id")
        if student_id is None:
            telegram_id = callback.from_user.id if callback.from_user else 0
            name = (
                callback.from_user.full_name
                if callback.from_user and callback.from_user.full_name
                else FALLBACK_NAME
            )
            username = callback.from_user.username if callback.from_user else None
            student = uow.students.get_or_create(
                telegram_id=telegram_id,
                name=name,
                username=username,
            )
            student_id = student.student_id
        option_id = uow.books.get_option_id(book_id, option_number)
        session = uow.sessions.create(student_id, option_id)
        qs = uow.questions.list_for_option(option_id, "A")
        qs += uow.questions.list_for_option(option_id, "B")
        return session.session_id, [(q.question_id, q.part) for q in qs]

    session_id, question_ids = await with_uow(db_path, start_test)
    await callback.message.edit_text(MSG_START_TESTING)
    await state.update_data(
        session_id=session_id,
        question_ids=question_ids,
        current_index=0,
    )
    await state.set_state(Testing.answering)
    await callback.answer()

    await send_current_question(callback.message, state, callback.bot, db_path)
