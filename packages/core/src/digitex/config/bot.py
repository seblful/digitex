"""Settings for the Telegram bot — the one workflow that deploys."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    """Telegram bot settings.

    Both fields default to a falsy value rather than being required, so
    importing the bot and running ``--help`` works on a machine with no
    credentials. The entry point checks the token and refuses to start polling
    without one.
    """

    model_config = SettingsConfigDict(env_prefix="BOT_", extra="ignore")

    token: str = Field(
        default="",
        description="Telegram bot token from @BotFather",
    )

    admin_user_id: int = Field(
        default=0,
        description="Telegram user ID of the bot admin who approves registrations",
    )
