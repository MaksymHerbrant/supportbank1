"""Routers build without Telegram (catches wiring errors in handlers)."""
from __future__ import annotations

from aiogram import Bot, Dispatcher

from app.bot.cards import TelegramNotifier
from app.bot.client_handlers import build_client_router
from app.bot.operator_handlers import build_operator_router
from tests.conftest import make_container


async def test_routers_build(db, notifier) -> None:  # noqa: ANN001
    bot = Bot("123456:TEST_token_for_wiring_only")
    tg = TelegramNotifier(bot, db, -100, -200)
    c = make_container(db, notifier, operator_chat_id=-100)
    dp = Dispatcher()
    dp.include_router(build_operator_router(c, tg))
    dp.include_router(build_client_router(c))
    assert set(dp.resolve_used_update_types()) >= {"message", "callback_query"}
    await bot.session.close()
