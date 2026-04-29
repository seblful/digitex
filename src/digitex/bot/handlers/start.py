"""Start command and student registration."""

from aiogram import Router, types
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext

from digitex.bot.database import with_uow
from digitex.bot.keyboards import subjects_kb
from digitex.bot.messages import MSG_GREETING
from digitex.bot.states import Navigation
from digitex.config import get_settings

router = Router()


@router.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext) -> None:
    db_path = get_settings().database.path
    fallback_name = "Пользователь"

    def register(uow):
        telegram_id = message.from_user.id if message.from_user else 0
        name = message.from_user.full_name if message.from_user else fallback_name
        username = message.from_user.username if message.from_user else None
        student = uow.students.get_or_create(
            telegram_id=telegram_id,
            name=name,
            username=username,
        )
        subjects = uow.books.list_subjects()
        return student, subjects

    student, subjects = await with_uow(db_path, register)
    user_name = message.from_user.full_name if message.from_user else fallback_name
    await state.update_data(student_id=student.student_id)
    await message.answer(
        MSG_GREETING.format(name=user_name),
        reply_markup=subjects_kb(subjects),
    )
    await state.set_state(Navigation.select_subject)
