"""Start command and student registration."""

from aiogram import Router, types
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext

from digitex.bot.database import with_uow
from digitex.bot.keyboards import subjects_kb
from digitex.bot.states import Navigation
from digitex.config import get_settings

router = Router()


@router.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext) -> None:
    db_path = get_settings().database.path

    def register(uow):
        telegram_id = message.from_user.id if message.from_user else 0
        name = message.from_user.full_name if message.from_user else "Пользователь"
        username = message.from_user.username if message.from_user else None
        uow.students.get_or_create(
            telegram_id=telegram_id,
            name=name,
            username=username,
        )
        rows = uow._conn.execute(
            "SELECT subject_id, name FROM subjects ORDER BY name"
        ).fetchall()
        return rows

    subjects = await with_uow(db_path, register)
    user_name = message.from_user.full_name if message.from_user else "Пользователь"
    await message.answer(
        f"Здравствуйте, {user_name}! Выберите предмет:",
        reply_markup=subjects_kb(subjects),
    )
    await state.set_state(Navigation.select_subject)
