"""Subject → Year → Option selection callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiogram import Router, types

from digitex.bot import fsm_data
from digitex.bot.callbacks import (
    ExamTypeCB,
    ModeCB,
    OptionCB,
    RandomPartCB,
    SubjectCB,
    TopicCB,
    YearCB,
)
from digitex.bot.constants import FALLBACK_NAME
from digitex.bot.fsm_data import NavigationState, TestingState
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
from digitex.core.db import UnitOfWork

if TYPE_CHECKING:
    from aiogram.fsm.context import FSMContext
    from psycopg_pool import AsyncConnectionPool

    from digitex.core.domain import ExamType

router = Router()


@router.callback_query(Navigation.select_subject, SubjectCB.filter())
async def on_subject_selected(
    callback: types.CallbackQuery,
    callback_data: SubjectCB,
    state: FSMContext,
) -> None:
    if not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    nav = await fsm_data.load(state, NavigationState)
    student_id = nav.student_id
    await state.clear()
    await fsm_data.save(
        state,
        NavigationState(subject_id=callback_data.subject_id, student_id=student_id),
    )
    await callback.message.edit_text(MSG_MODE_SELECT, reply_markup=mode_kb())
    await state.set_state(Navigation.select_mode)
    await callback.answer()


@router.callback_query(Navigation.select_mode, ModeCB.filter())
async def on_mode_selected(
    callback: types.CallbackQuery,
    callback_data: ModeCB,
    state: FSMContext,
    pool: AsyncConnectionPool,
) -> None:
    if not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    nav = await fsm_data.load(state, NavigationState)
    if nav.subject_id is None:
        await callback.answer()
        return
    subject_id = nav.subject_id

    match callback_data.mode:
        case "standard":
            async with UnitOfWork(pool) as uow:
                years = await uow.books.list_years(subject_id)
                if not years:
                    subjects = await uow.books.list_subjects()
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
            async with UnitOfWork(pool) as uow:
                topics = await uow.questions.get_topics_for_subject(subject_id)
            if not topics:
                await callback.message.edit_text(MSG_NO_TOPICS)
                await callback.answer()
                return
            await callback.message.edit_text(
                MSG_TOPIC_SELECT, reply_markup=topics_kb(topics)
            )
            await fsm_data.merge(state, topic_names=topics)
            await state.set_state(Navigation.select_topic)

    await callback.answer()


@router.callback_query(Navigation.select_topic, TopicCB.filter())
async def on_topic_selected(
    callback: types.CallbackQuery,
    callback_data: TopicCB,
    state: FSMContext,
    pool: AsyncConnectionPool,
) -> None:
    if not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    nav = await fsm_data.load(state, NavigationState)
    if not nav.topic_names:
        await callback.answer()
        return
    topic_name = nav.topic_names[callback_data.index]
    await fsm_data.merge(state, topic_name=topic_name)

    assert callback.bot is not None  # aiogram injects this on every callback
    await start_random_question(callback.message, state, callback.bot, pool)
    await callback.answer()


@router.callback_query(Navigation.select_random_exam_type, ExamTypeCB.filter())
async def on_random_exam_type_selected(
    callback: types.CallbackQuery,
    callback_data: ExamTypeCB,
    state: FSMContext,
) -> None:
    if not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    await fsm_data.merge(state, exam_type=callback_data.exam_type)
    await callback.message.edit_text(MSG_PART_SELECT, reply_markup=random_part_kb())
    await state.set_state(Navigation.select_random_part)
    await callback.answer()


@router.callback_query(Navigation.select_random_part, RandomPartCB.filter())
async def on_random_part_selected(
    callback: types.CallbackQuery,
    callback_data: RandomPartCB,
    state: FSMContext,
    pool: AsyncConnectionPool,
) -> None:
    if not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    await fsm_data.merge(state, random_part=callback_data.part)

    assert callback.bot is not None
    await start_random_question(callback.message, state, callback.bot, pool)
    await callback.answer()


@router.callback_query(Navigation.select_year, YearCB.filter())
async def on_year_selected(
    callback: types.CallbackQuery,
    callback_data: YearCB,
    state: FSMContext,
    pool: AsyncConnectionPool,
) -> None:
    if not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    year = callback_data.year
    await fsm_data.merge(state, year=year)

    if year >= 2023:
        await callback.message.edit_text(
            MSG_EXAM_TYPE_SELECT,
            reply_markup=exam_type_kb(),
        )
        await state.set_state(Navigation.select_exam_type)
    else:
        await _show_options_for_exam_type(callback.message, state, year, "CT", pool)

    await callback.answer()


@router.callback_query(Navigation.select_exam_type, ExamTypeCB.filter())
async def on_exam_type_selected(
    callback: types.CallbackQuery,
    callback_data: ExamTypeCB,
    state: FSMContext,
    pool: AsyncConnectionPool,
) -> None:
    if not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    nav = await fsm_data.load(state, NavigationState)
    if nav.year is None:
        await callback.answer()
        return
    await _show_options_for_exam_type(
        callback.message, state, nav.year, callback_data.exam_type, pool
    )
    await callback.answer()


async def _show_options_for_exam_type(
    message: types.Message,
    state: FSMContext,
    year: int,
    exam_type: ExamType,
    pool: AsyncConnectionPool,
) -> None:
    nav = await fsm_data.load(state, NavigationState)
    if nav.subject_id is None:
        return

    async with UnitOfWork(pool) as uow:
        book_id = await uow.books.get_book(nav.subject_id, year)
        options = await uow.books.list_options(book_id, exam_type) if book_id else []

    if not options:
        await message.edit_text(MSG_NO_OPTIONS.format(exam_type=exam_type))
        await state.set_state(Navigation.select_year)
        return

    await message.edit_text(MSG_OPTION_SELECT, reply_markup=options_kb(options))
    await fsm_data.merge(state, book_id=book_id, exam_type=exam_type)
    await state.set_state(Navigation.select_option)


@router.callback_query(Navigation.select_option, OptionCB.filter())
async def on_option_selected(
    callback: types.CallbackQuery,
    callback_data: OptionCB,
    state: FSMContext,
    pool: AsyncConnectionPool,
) -> None:
    if not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    nav = await fsm_data.load(state, NavigationState)
    if nav.book_id is None:
        await callback.answer()
        return
    book_id = nav.book_id
    student_id = nav.student_id

    async with UnitOfWork(pool) as uow:
        if student_id is None:
            telegram_id = callback.from_user.id if callback.from_user else 0
            name = (
                callback.from_user.full_name
                if callback.from_user and callback.from_user.full_name
                else FALLBACK_NAME
            )
            username = callback.from_user.username if callback.from_user else None
            student = await uow.students.get_or_create(
                telegram_id=telegram_id,
                name=name,
                username=username,
            )
            student_id = student.student_id
        option_id = await uow.books.get_option_id(book_id, callback_data.option)
        session = await uow.sessions.create(student_id, option_id)
        question_ids = await uow.questions.list_ids_for_option(option_id)
        session_id = session.session_id

    await callback.message.edit_text(MSG_START_TESTING)
    await fsm_data.save(
        state,
        TestingState(
            student_id=student_id,
            session_id=session_id,
            question_ids=question_ids,
            current_index=0,
        ),
    )
    await state.set_state(Testing.answering)
    await callback.answer()

    assert callback.bot is not None
    await send_current_question(callback.message, state, callback.bot, pool)
