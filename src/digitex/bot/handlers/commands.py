"""Global bot command handlers."""

from aiogram import Router, types
from aiogram.filters import Command

from digitex.bot.messages import MSG_HELP

router = Router()


@router.message(Command("help"))
async def cmd_help(message: types.Message) -> None:
    await message.answer(MSG_HELP, parse_mode="HTML")
