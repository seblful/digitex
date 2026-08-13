"""Start command, registration flow, and admin approval callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from aiogram import Bot, Router, types
from aiogram.filters import Command, CommandStart
from aiogram.types import Message as TgMessage
from aiogram.utils.text_decorations import html_decoration

from digitex.bot import fsm_data
from digitex.bot.answer_flow import end_round
from digitex.bot.callbacks import RegistrationCB
from digitex.bot.constants import student_identity
from digitex.bot.keyboards import admin_registration_kb, subjects_kb
from digitex.bot.messages import (
    MSG_ADMIN_NEW_REQUEST,
    MSG_ADMIN_ONLY,
    MSG_APPROVED_ADMIN,
    MSG_APPROVED_USER,
    MSG_ASK_NAME,
    MSG_GREETING,
    MSG_HELP,
    MSG_PENDING,
    MSG_REGISTRATION_INFO,
    MSG_REJECTED_ADMIN,
    MSG_REJECTED_USER,
    MSG_REQUEST_SENT,
)
from digitex.bot.states import Navigation, Registration
from digitex.db import UnitOfWork

if TYPE_CHECKING:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from aiogram.fsm.context import FSMContext
    from psycopg_pool import AsyncConnectionPool

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


def _format_datetime(dt: datetime, tz: ZoneInfo) -> str:
    local = dt.astimezone(tz)
    time_str = f"{local.hour:02d}:{local.minute:02d}"
    return f"{local.day} {MONTHS_RU[local.month - 1]} {local.year} в {time_str}"


@dataclass(frozen=True)
class StartGate:
    """Whether /start proceeds to subject selection, and what to show if not.

    ``new`` covers both a first-time user and one whose earlier request was
    rejected — /start treats them identically, by asking for a name.
    """

    status: Literal["new", "pending", "approved"]
    requested_at: datetime | None = None


async def open_registration_gate(uow: UnitOfWork, telegram_id: int) -> StartGate:
    """Read the user's registration record.

    One round-trip, and a pure read: the student row carries both the status and
    the submission date. A rejected student counts as new because re-applying
    overwrites the decision — nothing has to be deleted to let them back in, so
    the rejection stays on the record until they actually reapply.
    """
    student = await uow.students.get(telegram_id)

    if student is None or student.status == "rejected":
        return StartGate(status="new")
    if student.status == "pending":
        return StartGate(status="pending", requested_at=student.created_at)
    return StartGate(status="approved")


async def _normal_start(
    message: types.Message, state: FSMContext, pool: AsyncConnectionPool
) -> None:
    telegram_id, name, username = student_identity(message)

    async with UnitOfWork(pool) as uow:
        student = await uow.students.get_or_create(
            telegram_id=telegram_id,
            telegram_name=name,
            telegram_username=username,
        )
        subjects = await uow.books.list_subjects()

    # /start can land mid-test, where the last render may still owe a file_id
    # write; ending the round pays it before the state goes away.
    await end_round(pool, state)
    await fsm_data.merge(state, student_telegram_id=student.telegram_id)
    await message.answer(
        MSG_GREETING.format(name=name),
        reply_markup=subjects_kb(subjects),
    )
    await state.set_state(Navigation.select_subject)


@router.message(Command("help"))
async def cmd_help(message: types.Message) -> None:
    await message.answer(MSG_HELP, parse_mode="HTML")


@router.message(CommandStart())
async def cmd_start(
    message: types.Message,
    state: FSMContext,
    pool: AsyncConnectionPool,
    admin_user_id: int,
    tz: ZoneInfo,
) -> None:
    telegram_id, _name, _username = student_identity(message)

    if telegram_id == admin_user_id:
        await _normal_start(message, state, pool)
        return

    async with UnitOfWork(pool) as uow:
        gate = await open_registration_gate(uow, telegram_id)

    if gate.status == "approved":
        await _normal_start(message, state, pool)
        return

    if gate.status == "pending":
        date_str = _format_datetime(gate.requested_at, tz) if gate.requested_at else "—"
        await message.answer(MSG_PENDING.format(date=date_str), parse_mode="HTML")
        return

    await state.set_state(Registration.waiting_for_name)
    await message.answer(MSG_REGISTRATION_INFO, parse_mode="HTML")
    await message.answer(MSG_ASK_NAME, parse_mode="HTML")


@router.message(Registration.waiting_for_name)
async def process_name(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    pool: AsyncConnectionPool,
    admin_user_id: int,
    tz: ZoneInfo,
) -> None:
    telegram_id, telegram_name, username = student_identity(message)
    full_name = (message.text or "").strip()

    if not full_name:
        await message.answer(MSG_ASK_NAME, parse_mode="HTML")
        return

    async with UnitOfWork(pool) as uow:
        request = await uow.students.create_request(
            telegram_id=telegram_id,
            full_name=full_name,
            telegram_name=telegram_name,
            telegram_username=username,
        )
    await state.clear()

    # The name is whatever the user typed, and both messages are parsed as
    # HTML — the admin's copy included.
    safe_name = html_decoration.quote(full_name)
    date_str = _format_datetime(request.created_at, tz)
    await message.answer(
        MSG_REQUEST_SENT.format(name=safe_name, date=date_str),
        parse_mode="HTML",
    )

    await bot.send_message(
        admin_user_id,
        MSG_ADMIN_NEW_REQUEST.format(
            full_name=safe_name,
            username=html_decoration.quote(username) if username else "—",
            telegram_id=telegram_id,
        ),
        parse_mode="HTML",
        reply_markup=admin_registration_kb(telegram_id),
    )


@router.callback_query(RegistrationCB.filter())
async def handle_reg_callback(
    callback: types.CallbackQuery,
    callback_data: RegistrationCB,
    bot: Bot,
    pool: AsyncConnectionPool,
    admin_user_id: int,
) -> None:
    if callback.from_user.id != admin_user_id:
        await callback.answer(MSG_ADMIN_ONLY, show_alert=True)
        return

    target_id = callback_data.telegram_id
    admin_id, admin_name, admin_username = student_identity(callback)

    async with UnitOfWork(pool) as uow:
        # A decision names the student who made it, so the admin needs a row of
        # their own before it can be recorded against them.
        await uow.students.get_or_create(
            telegram_id=admin_id,
            telegram_name=admin_name,
            telegram_username=admin_username,
        )
        if callback_data.action == "approve":
            student = await uow.students.approve(target_id, admin_id)
            user_message = MSG_APPROVED_USER
            admin_reply = MSG_APPROVED_ADMIN.format(full_name=student.full_name)
        else:
            student = await uow.students.reject(target_id, admin_id)
            user_message = MSG_REJECTED_USER
            admin_reply = MSG_REJECTED_ADMIN.format(full_name=student.full_name)

    await bot.send_message(target_id, user_message)
    if isinstance(callback.message, TgMessage):
        await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer(admin_reply)
