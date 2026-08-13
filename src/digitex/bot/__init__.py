"""Telegram bot package.

Deliberately empty of imports: the handlers reach back for
``digitex.bot.fsm_data``, so anything imported here would run on the way into
every one of them. The dispatcher is assembled in :mod:`digitex.bot.dispatcher`.
"""
