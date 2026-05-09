"""Start command, registration flow, and admin approval callbacks."""

from datetime import UTC, datetime

from aiogram import Bot, Router, types
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message as TgMessage

from digitex.bot.constants import FALLBACK_NAME
from digitex.bot.database import with_uow
from digitex.bot.keyboards import admin_registration_kb, subjects_kb
from digitex.bot.messages import (
    MSG_ADMIN_NEW_REQUEST,
    MSG_APPROVED_ADMIN,
    MSG_APPROVED_USER,
    MSG_ASK_NAME,
    MSG_GREETING,
    MSG_PENDING,
    MSG_REGISTRATION_INFO,
    MSG_REJECTED_ADMIN,
    MSG_REJECTED_USER,
    MSG_REQUEST_SENT,
)
from digitex.bot.states import Navigation, Registration
from digitex.utils import get_tz

router = Router()

MONTHS_RU = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]


def _format_datetime(dt: datetime) -> str:
    local = dt.replace(tzinfo=UTC).astimezone(get_tz())
    time_str = f"{local.hour:02d}:{local.minute:02d}"
    return f"{local.day} {MONTHS_RU[local.month - 1]} {local.year} в {time_str}"


def _get_user_info(
    message: types.Message,
) -> tuple[int, str, str | None]:
    user = message.from_user
    if user:
        return user.id, user.full_name or FALLBACK_NAME, user.username
    return 0, FALLBACK_NAME, None


async def _normal_start(
    message: types.Message, state: FSMContext, db_path: str
) -> None:
    def register(uow):
        telegram_id = message.from_user.id if message.from_user else 0
        name = message.from_user.full_name if message.from_user else FALLBACK_NAME
        username = message.from_user.username if message.from_user else None
        student = uow.students.get_or_create(
            telegram_id=telegram_id,
            name=name,
            username=username,
        )
        subjects = uow.books.list_subjects()
        return student, subjects

    student, subjects = await with_uow(db_path, register)
    user_name = message.from_user.full_name if message.from_user else FALLBACK_NAME
    await state.clear()
    await state.update_data(student_id=student.student_id)
    await message.answer(
        MSG_GREETING.format(name=user_name),
        reply_markup=subjects_kb(subjects),
    )
    await state.set_state(Navigation.select_subject)


@router.message(CommandStart())
async def cmd_start(
    message: types.Message, state: FSMContext, db_path: str, admin_user_id: int
) -> None:
    telegram_id, _name, _username = _get_user_info(message)

    if telegram_id == admin_user_id:
        await _normal_start(message, state, db_path)
        return

    status = await with_uow(
        db_path, lambda uow: uow.authorized_users.get_status(telegram_id)
    )

    if status is None:
        await state.set_state(Registration.waiting_for_name)
        await message.answer(MSG_REGISTRATION_INFO, parse_mode="HTML")
        await message.answer(MSG_ASK_NAME, parse_mode="HTML")
    elif status == "pending":
        request = await with_uow(
            db_path, lambda uow: uow.authorized_users.get_request(telegram_id)
        )
        date_str = _format_datetime(request.created_at) if request else "—"
        await message.answer(MSG_PENDING.format(date=date_str), parse_mode="HTML")
    elif status == "rejected":
        await with_uow(
            db_path, lambda uow: uow.authorized_users.delete_request(telegram_id)
        )
        await state.set_state(Registration.waiting_for_name)
        await message.answer(MSG_REGISTRATION_INFO, parse_mode="HTML")
        await message.answer(MSG_ASK_NAME, parse_mode="HTML")
    else:
        await _normal_start(message, state, db_path)


@router.message(Registration.waiting_for_name)
async def process_name(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    db_path: str,
    admin_user_id: int,
) -> None:
    telegram_id, _, username = _get_user_info(message)
    full_name = (message.text or "").strip()

    if not full_name:
        await message.answer("Пожалуйста, введите ваши имя и фамилию:")
        return

    def create(uow):
        return uow.authorized_users.create_request(
            telegram_id=telegram_id,
            full_name=full_name,
            telegram_username=username,
        )

    request = await with_uow(db_path, create)
    await state.clear()

    date_str = _format_datetime(request.created_at)
    await message.answer(
        MSG_REQUEST_SENT.format(name=full_name, date=date_str),
        parse_mode="HTML",
    )

    await bot.send_message(
        admin_user_id,
        MSG_ADMIN_NEW_REQUEST.format(
            full_name=full_name,
            username=username or "—",
            telegram_id=telegram_id,
        ),
        parse_mode="HTML",
        reply_markup=admin_registration_kb(telegram_id),
    )


@router.callback_query(lambda c: c.data and c.data.startswith("reg:"))
async def handle_reg_callback(
    callback: types.CallbackQuery, bot: Bot, db_path: str, admin_user_id: int
) -> None:
    if callback.from_user.id != admin_user_id:
        await callback.answer(
            "Только администратор может выполнять это действие.", show_alert=True
        )
        return

    if not callback.data:
        await callback.answer("Некорректные данные.")
        return

    parts = callback.data.split(":")
    if len(parts) != 3:
        await callback.answer("Некорректные данные.")
        return

    action = parts[1]
    try:
        target_id = int(parts[2])
    except ValueError:
        await callback.answer("Некорректный ID пользователя.")
        return

    if action == "approve":

        def approve(uow):
            return uow.authorized_users.approve(target_id, callback.from_user.id)

        user_record = await with_uow(db_path, approve)
        await bot.send_message(target_id, MSG_APPROVED_USER)
        if isinstance(callback.message, TgMessage):
            await callback.message.edit_reply_markup(reply_markup=None)
        await callback.answer(
            MSG_APPROVED_ADMIN.format(full_name=user_record.full_name)
        )
    elif action == "reject":

        def reject(uow):
            return uow.authorized_users.reject(target_id, callback.from_user.id)

        user_record = await with_uow(db_path, reject)
        await bot.send_message(target_id, MSG_REJECTED_USER)
        if isinstance(callback.message, TgMessage):
            await callback.message.edit_reply_markup(reply_markup=None)
        await callback.answer(
            MSG_REJECTED_ADMIN.format(full_name=user_record.full_name)
        )
    else:
        await callback.answer("Неизвестное действие.")
