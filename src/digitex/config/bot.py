"""Settings for the Telegram bot — the one workflow that deploys."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    """Telegram bot settings."""

    model_config = SettingsConfigDict(env_prefix="BOT_", extra="ignore")

    token: str = Field(
        default="",
        description="Telegram bot token from @BotFather",
    )

    admin_user_id: int = Field(
        default=0,
        description="Telegram user ID of the bot admin who approves registrations",
    )
