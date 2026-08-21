"""The walk down to a question set: subject → mode → year → exam type → option.

Every screen here edits the message the tap came from, so the student sees one
keyboard replaced by the next rather than a growing column of them. Each
handler re-reads the navigation state it needs and answers the tap without
acting when a step is missing — a keyboard from before a restart is still live
in the chat.

Two of these screens are where a question round begins, which is why some
handlers here take the injected :class:`Round`: the mode chosen here decides
which loop the conversation enters.
"""

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
from digitex.bot.constants import student_identity
from digitex.bot.fsm_data import NavigationState, RandomState, TestingState
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
from digitex.domain.entities import year_has_exam_types

if TYPE_CHECKING:
    from aiogram.fsm.context import FSMContext

    from digitex.bot.answer_flow import Round
    from digitex.domain.entities import ExamType
    from digitex.domain.ports import OpenUow

router = Router()


@router.callback_query(Navigation.select_subject, SubjectCB.filter())
async def on_subject_selected(
    callback: types.CallbackQuery,
    callback_data: SubjectCB,
    state: FSMContext,
    msg: types.Message,
) -> None:
    # ``save`` replaces the whole data dict — a fresh navigation from the
    # subject down. This is a navigation step, not the end of a round, so no
    # clear() and no debt to pay.
    await fsm_data.save(state, NavigationState(subject_id=callback_data.subject_id))
    await msg.edit_text(MSG_MODE_SELECT, reply_markup=mode_kb())
    await state.set_state(Navigation.select_mode)
    await callback.answer()


@router.callback_query(Navigation.select_mode, ModeCB.filter())
async def on_mode_selected(
    callback: types.CallbackQuery,
    callback_data: ModeCB,
    state: FSMContext,
    msg: types.Message,
    open_uow: OpenUow,
) -> None:
    nav = await fsm_data.load(state, NavigationState)
    if nav.subject_id is None:
        await callback.answer()
        return
    subject_id = nav.subject_id

    match callback_data.mode:
        case "standard":
            # Read everything first: a Telegram round-trip inside the
            # transaction would hold it open past the statement timeout.
            async with open_uow() as uow:
                years = await uow.books.list_years(subject_id)
                subjects = [] if years else await uow.books.list_subjects()

            if years:
                await msg.edit_text(MSG_YEAR_SELECT, reply_markup=years_kb(years))
                await state.set_state(Navigation.select_year)
            else:
                # A subject nothing was extracted for yet: back to the subject
                # list rather than a screen with no way on.
                await msg.edit_text(MSG_NO_YEARS, reply_markup=subjects_kb(subjects))
                await state.set_state(Navigation.select_subject)

        case "random":
            await msg.edit_text(MSG_EXAM_TYPE_SELECT, reply_markup=exam_type_kb())
            await state.set_state(Navigation.select_random_exam_type)

        case "topics":
            async with open_uow() as uow:
                topics = await uow.topics.get_topics_for_subject(subject_id)

            if topics:
                await msg.edit_text(MSG_TOPIC_SELECT, reply_markup=topics_kb(topics))
                # The names are kept because a topic button carries its
                # position, not its name — the payload is too small for one.
                await fsm_data.merge(state, NavigationState, topic_names=topics)
                await state.set_state(Navigation.select_topic)
            else:
                # The mode screen's own keyboard is gone with the edit, but the
                # state stays put, so /start is the way on.
                await msg.edit_text(MSG_NO_TOPICS)

    await callback.answer()


async def _begin_random_round(
    message: types.Message, round: Round, rnd: RandomState
) -> None:
    """Enter random / topic mode with its state built whole.

    Built whole, the way the testing loop builds ``TestingState``, rather than
    accumulated field by field: the round needs a subject, and a constructor
    argument is what makes that a requirement instead of a hope.
    """
    await round.save(rnd)
    await start_random_question(message, round)


@router.callback_query(Navigation.select_topic, TopicCB.filter())
async def on_topic_selected(
    callback: types.CallbackQuery,
    callback_data: TopicCB,
    state: FSMContext,
    msg: types.Message,
    round: Round,
) -> None:
    nav = await fsm_data.load(state, NavigationState)
    # The index comes off a keyboard that may have been built for a different
    # subject's (longer) topic list.
    if (
        nav.subject_id is None
        or not nav.topic_names
        or not 0 <= callback_data.index < len(nav.topic_names)
    ):
        await callback.answer()
        return

    await _begin_random_round(
        msg,
        round,
        RandomState(
            subject_id=nav.subject_id,
            topic_name=nav.topic_names[callback_data.index],
        ),
    )
    await callback.answer()


@router.callback_query(Navigation.select_random_exam_type, ExamTypeCB.filter())
async def on_random_exam_type_selected(
    callback: types.CallbackQuery,
    callback_data: ExamTypeCB,
    state: FSMContext,
    msg: types.Message,
) -> None:
    await fsm_data.merge(state, NavigationState, exam_type=callback_data.exam_type)
    await msg.edit_text(MSG_PART_SELECT, reply_markup=random_part_kb())
    await state.set_state(Navigation.select_random_part)
    await callback.answer()


@router.callback_query(Navigation.select_random_part, RandomPartCB.filter())
async def on_random_part_selected(
    callback: types.CallbackQuery,
    callback_data: RandomPartCB,
    state: FSMContext,
    msg: types.Message,
    round: Round,
) -> None:
    nav = await fsm_data.load(state, NavigationState)
    if nav.subject_id is None:
        await callback.answer()
        return

    await _begin_random_round(
        msg,
        round,
        RandomState(
            subject_id=nav.subject_id,
            exam_type=nav.exam_type,
            random_part=callback_data.part,
        ),
    )
    await callback.answer()


@router.callback_query(Navigation.select_year, YearCB.filter())
async def on_year_selected(
    callback: types.CallbackQuery,
    callback_data: YearCB,
    state: FSMContext,
    msg: types.Message,
    open_uow: OpenUow,
) -> None:
    # Read before the write: the year is what changes, and the helper below
    # only needs the subject the navigation started from.
    nav = await fsm_data.load(state, NavigationState)
    year = callback_data.year
    await fsm_data.merge(state, NavigationState, year=year)

    if year_has_exam_types(year):
        await msg.edit_text(
            MSG_EXAM_TYPE_SELECT,
            reply_markup=exam_type_kb(),
        )
        await state.set_state(Navigation.select_exam_type)
    else:
        # Books before 2023 come in one flavour, so asking would offer a choice
        # that does not exist.
        await _show_options_for_exam_type(msg, state, nav, year, "CT", open_uow)

    await callback.answer()


@router.callback_query(Navigation.select_exam_type, ExamTypeCB.filter())
async def on_exam_type_selected(
    callback: types.CallbackQuery,
    callback_data: ExamTypeCB,
    state: FSMContext,
    msg: types.Message,
    open_uow: OpenUow,
) -> None:
    nav = await fsm_data.load(state, NavigationState)
    if nav.year is None:
        await callback.answer()
        return
    await _show_options_for_exam_type(
        msg, state, nav, nav.year, callback_data.exam_type, open_uow
    )
    await callback.answer()


async def _show_options_for_exam_type(
    message: types.Message,
    state: FSMContext,
    nav: NavigationState,
    year: int,
    exam_type: ExamType,
    open_uow: OpenUow,
) -> None:
    """Offer the options one book holds for *exam_type*, or send the year back.

    Reached from two screens — a pre-2023 year, where the exam type is implied,
    and the exam-type keyboard itself — so *year* and *exam_type* are arguments
    rather than state reads.
    """
    if nav.subject_id is None:
        return

    async with open_uow() as uow:
        book_id = await uow.books.get_book(nav.subject_id, year)
        options = await uow.books.list_options(book_id, exam_type) if book_id else []

    if not options:
        await message.edit_text(MSG_NO_OPTIONS.format(exam_type=exam_type))
        await state.set_state(Navigation.select_year)
        return

    await message.edit_text(MSG_OPTION_SELECT, reply_markup=options_kb(options))
    await fsm_data.merge(state, NavigationState, book_id=book_id, exam_type=exam_type)
    await state.set_state(Navigation.select_option)


@router.callback_query(Navigation.select_option, OptionCB.filter())
async def on_option_selected(
    callback: types.CallbackQuery,
    callback_data: OptionCB,
    state: FSMContext,
    msg: types.Message,
    round: Round,
) -> None:
    nav = await fsm_data.load(state, NavigationState)
    if nav.book_id is None:
        await callback.answer()
        return
    book_id = nav.book_id

    async with round.transaction() as uow:
        # The session references the student, so the row has to exist — and
        # the tap in hand says who is asking. Deriving the identity from the
        # event every time also covers a tap that arrives after the FSM was
        # lost to a restart.
        telegram_id, name, username = student_identity(callback)
        student = await uow.students.get_or_create(
            telegram_id=telegram_id,
            telegram_name=name,
            telegram_username=username,
        )
        option_id = await uow.books.get_option_id(book_id, callback_data.option)
        session = await uow.sessions.create(student.telegram_id, option_id)
        question_ids = await uow.questions.list_ids_for_option(option_id)
        session_id = session.session_id

    await msg.edit_text(MSG_START_TESTING)
    await fsm_data.save(
        state,
        TestingState(
            session_id=session_id,
            question_ids=question_ids,
            current_index=0,
        ),
    )
    await state.set_state(Testing.answering)
    await callback.answer()

    # Acknowledged first: the first render uploads an image, and Telegram would
    # otherwise leave the tap spinning for the length of that upload.
    await send_current_question(msg, round)
