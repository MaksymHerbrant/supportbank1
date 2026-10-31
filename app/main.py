"""Entry point: ``python -m app.main`` (Telegram long polling).

``python -m app.main --check`` validates configuration, DB, LLM and embeddings without Telegram.
"""
from __future__ import annotations

import asyncio
import logging
import sys

from app.config import get_settings
from app.container import build
from app.db.repo import Database
from app.db.seed import seed
from app.notify import RecordingNotifier

log = logging.getLogger("app")
REFRESH_EVERY_S = 60


async def _refresh_loop(container, notifier) -> None:  # noqa: ANN001
    """Periodically re-check open incidents: at_risk -> affected promotions and live card timers."""
    while True:
        await asyncio.sleep(REFRESH_EVERY_S)
        try:
            await container.incidents.refresh_all_open()
            for row in container.db.q("SELECT id FROM incidents WHERE status='open'"):
                await notifier.incident_changed(row["id"])
        except Exception:  # noqa: BLE001
            log.exception("refresh loop iteration failed")


def _open_db(path: str) -> Database:
    db = Database(path)
    if not db.has_schema():
        log.info("Empty database, seeding demo data")
        seed(db)
    return db


async def check() -> int:
    settings = get_settings()
    db = _open_db(settings.db_path)
    c = build(settings, db, RecordingNotifier())
    print(f"DB: {settings.db_path} · customers={db.scalar('SELECT COUNT(*) FROM customers')}")
    print(f"LLM backend: {settings.llm_backend} · available={c.llm.available}")
    if c.llm.available:
        out = await c.llm.text("Reply with one word.", "Say OK")
        print(f"LLM ping: {'ok' if out else 'FAILED (rule-based fallback will be used)'}")
    vec = await c.store.embedder.embed(["перевірка"])
    print(f"Embeddings: {'ok, dim=' + str(len(vec[0])) if vec else 'unavailable (exact-signature mode)'}")
    print(f"FAQ chunks: {len(c.faq.chunks)} · embedded={await c.faq.index()}")
    print(f"Telegram token set: {bool(settings.bot_token)} · operator chat: {settings.operator_chat_id} · "
          f"dev chat: {settings.dev_chat_id}")
    db.close()
    return 0


async def run() -> int:
    from aiogram import Bot, Dispatcher
    from aiogram.client.default import DefaultBotProperties

    from app.bot.cards import TelegramNotifier
    from app.bot.client_handlers import build_client_router
    from app.bot.operator_handlers import build_operator_router

    settings = get_settings()
    if not settings.bot_token or ":" not in settings.bot_token:
        log.error("BOT_TOKEN is not set. Copy .env.example to .env and fill it in (or run with --check).")
        return 2
    db = _open_db(settings.db_path)
    bot = Bot(settings.bot_token, default=DefaultBotProperties(link_preview_is_disabled=True))
    notifier = TelegramNotifier(bot, db, settings.operator_chat_id, settings.dev_chat_id)
    container = build(settings, db, notifier)
    await container.faq.index()
    dp = Dispatcher()
    dp.include_router(build_operator_router(container, notifier))
    dp.include_router(build_client_router(container))
    refresher = asyncio.create_task(_refresh_loop(container, notifier))
    me = await bot.get_me()
    log.info("Bot @%s started · LLM=%s(%s) · operator chat=%s · dev chat=%s", me.username, settings.llm_backend,
             "on" if container.llm.available else "off", settings.operator_chat_id, settings.dev_chat_id)
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        refresher.cancel()
        await bot.session.close()
        db.close()
    return 0


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    coro = check() if "--check" in sys.argv else run()
    try:
        sys.exit(asyncio.run(coro))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
