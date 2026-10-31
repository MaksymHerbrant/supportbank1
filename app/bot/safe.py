"""Telegram helpers that never abort a handler because of a late/expired callback."""
from __future__ import annotations

import logging

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery

log = logging.getLogger(__name__)


async def answer_cb(cb: CallbackQuery, text: str | None = None, show_alert: bool = False) -> None:
    """Answer a button press; if Telegram says it is too old (network hiccup), keep going with the action."""
    try:
        await cb.answer(text, show_alert=show_alert)
    except TelegramBadRequest as exc:
        log.info("late callback answer ignored: %s", exc)
