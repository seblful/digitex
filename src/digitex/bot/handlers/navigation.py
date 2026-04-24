"""Subject → Year → Option selection callbacks."""

from aiogram import F, Router, types
from aiogram.fsm.context import FSMContext

from digitex.bot.database import with_uow
from digitex.bot.keyboards import options_kb, years_kb
from digitex.bot.states import Navigation, Testing
from digitex.config import get_settings

router = Router()


@router.callback_query(Navigation.select_subject, F.data.startswith("subj:"))
async def on_subject_selected(callback: types.CallbackQuery, state: FSMContext) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    subject_id = int(callback.data.split(":")[1])
    db_path = get_settings().database.path

    def fetch_years(uow):
        rows = uow._conn.execute(
            "SELECT year_value FROM books WHERE subject_id = ? ORDER BY year_value DESC",
            (subject_id,),
        ).fetchall()
        return [r[0] for r in rows]

    years = await with_uow(db_path, fetch_years)
    if not years:
        await callback.message.edit_text("No years available for this subject.")
        await callback.answer()
        return

    await callback.message.edit_text(
        "Select year:",
        reply_markup=years_kb(years),
    )
    await state.update_data(subject_id=subject_id)
    await state.set_state(Navigation.select_year)
    await callback.answer()


@router.callback_query(Navigation.select_year, F.data.startswith("year:"))
async def on_year_selected(callback: types.CallbackQuery, state: FSMContext) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    year = int(callback.data.split(":")[1])
    data = await state.get_data()
    subject_id = data["subject_id"]
    db_path = get_settings().database.path

    def fetch_options(uow):
        book_id = uow.books.get_or_create_book(subject_id, year)
        rows = uow._conn.execute(
            "SELECT option_number FROM options WHERE book_id = ? ORDER BY option_number",
            (book_id,),
        ).fetchall()
        return book_id, [r[0] for r in rows]

    book_id, options = await with_uow(db_path, fetch_options)
    if not options:
        await callback.message.edit_text("No options available for this year.")
        await callback.answer()
        return

    await callback.message.edit_text(
        "Select option:",
        reply_markup=options_kb(options),
    )
    await state.update_data(book_id=book_id)
    await state.set_state(Navigation.select_option)
    await callback.answer()


@router.callback_query(Navigation.select_option, F.data.startswith("opt:"))
async def on_option_selected(callback: types.CallbackQuery, state: FSMContext) -> None:
    if not callback.data or not isinstance(callback.message, types.Message):
        await callback.answer()
        return

    option_number = int(callback.data.split(":")[1])
    data = await state.get_data()
    book_id = data["book_id"]
    db_path = get_settings().database.path

    def start_test(uow):
        telegram_id = callback.from_user.id if callback.from_user else 0
        name = callback.from_user.full_name if callback.from_user and callback.from_user.full_name else "User"
        username = callback.from_user.username if callback.from_user else None
        student = uow.students.get_or_create(
            telegram_id=telegram_id,
            name=name,
            username=username,
        )
        session = uow.sessions.create(student.student_id, book_id, option_number)
        option_id = uow._conn.execute(
            "SELECT option_id FROM options WHERE book_id = ? AND option_number = ?",
            (book_id, option_number),
        ).fetchone()[0]
        qs = uow.questions.list_for_option(option_id, "A")
        qs += uow.questions.list_for_option(option_id, "B")
        return session.session_id, [q.question_id for q in qs]

    session_id, question_ids = await with_uow(db_path, start_test)
    await callback.message.edit_text("Starting test!")
    await state.update_data(
        session_id=session_id,
        question_ids=question_ids,
        current_index=0,
    )
    await state.set_state(Testing.answering)
    await callback.answer()

    from digitex.bot.handlers.testing import send_current_question
    await send_current_question(callback.message, state, callback.bot)
