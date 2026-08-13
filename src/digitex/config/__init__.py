"""Settings, one module per layer that reads them.

The facade: import from ``digitex.config`` rather than from the modules
underneath, so a group can move between them without touching callers.

    from digitex.config import get_settings

Nothing here imports a layer — settings are resolved at an entry point and
threaded down, never reached for deep in a call stack or at module import.
"""

from digitex.config.bot import BotSettings
from digitex.config.database import DatabaseSettings
from digitex.config.paths import PathsSettings
from digitex.config.pipeline import (
    DataSettings,
    ExtractionSettings,
    LabelStudioSettings,
    OpenRouterSettings,
    PipelineSettings,
)
from digitex.config.runtime import (
    AppSettings,
    LoggingSettings,
    LogLevel,
    TimezoneSettings,
)
from digitex.config.settings import Settings, get_settings, reset_settings_cache

__all__ = [
    "AppSettings",
    "BotSettings",
    "DataSettings",
    "DatabaseSettings",
    "ExtractionSettings",
    "LabelStudioSettings",
    "LogLevel",
    "LoggingSettings",
    "OpenRouterSettings",
    "PathsSettings",
    "PipelineSettings",
    "Settings",
    "TimezoneSettings",
    "get_settings",
    "reset_settings_cache",
]
