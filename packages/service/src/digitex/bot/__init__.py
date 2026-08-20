"""Telegram bot package — the conversation layer and nothing beneath it.

Kept free of imports on purpose. Every handler reaches back for
:mod:`digitex.bot.fsm_data`, which executes this module first, so anything
wired up here would run on the way into all of them and importing one handler
would drag in every other. The dispatcher is assembled in
:mod:`digitex.bot.dispatcher` for exactly that reason.
"""
