"""Customer bot: text, photos, videos. Media is processed in memory and never stored."""
from __future__ import annotations

import io
import logging

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.agent.media import MAX_VIDEO_BYTES, MediaError
from app.container import Container

log = logging.getLogger(__name__)

# Demo personas a real Telegram user can "become" (all data is fictional).
PERSONAS = [
    ("fuel_late_affected", "⛽ Клієнт з подвійним списанням Fuel+"),
    ("fuel_late_clean", "⛽ Клієнт Fuel+ без дубля"),
    ("phish_reporter_3", "🎣 Отримав фішингову SMS"),
    ("atm_reporter_3", "🏧 Банкомат зʼїв картку"),
    ("app_reporter_3", "📱 Не проходить оплата в застосунку"),
    ("fake_screenshot", "🧾 «Підроблений» скріншот"),
    ("social_eng", "📞 Продиктував код «банку»"),
    ("faq_user", "❓ Питання з FAQ"),
]
DEFAULT_PERSONA = "fuel_late_affected"


def build_client_router(c: Container) -> Router:
    router = Router(name="clients")
    router.message.filter(F.chat.type == "private")

    def persona_keyboard() -> InlineKeyboardMarkup:
        rows = []
        for tag, label in PERSONAS:
            cid = c.db.scalar("SELECT id FROM customers WHERE demo_tag=?", (tag,))
            if cid:
                rows.append([InlineKeyboardButton(text=label, callback_data=f"persona:{cid}")])
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def customer_for(chat_id: int) -> int:
        cid = c.db.scalar("SELECT id FROM customers WHERE tg_chat_id=?", (chat_id,))
        if cid:
            return int(cid)
        cid = c.db.scalar("SELECT id FROM customers WHERE demo_tag=?", (DEFAULT_PERSONA,))
        c.db.ex("UPDATE customers SET tg_chat_id=NULL WHERE tg_chat_id=?", (chat_id,))
        c.db.ex("UPDATE customers SET tg_chat_id=? WHERE id=?", (chat_id, cid))
        return int(cid)

    @router.message(CommandStart())
    @router.message(Command("persona"))
    async def start(message: Message) -> None:
        await message.answer(
            "Вітаю! Я — AI-асистент підтримки DemoBank (демо, усі дані вигадані).\n"
            "Оберіть демо-профіль клієнта, від імені якого будете писати:", reply_markup=persona_keyboard())

    @router.callback_query(F.data.startswith("persona:"))
    async def choose(cb: CallbackQuery) -> None:
        cid = int(cb.data.split(":")[1])
        chat_id = cb.message.chat.id if cb.message else cb.from_user.id
        c.db.ex("UPDATE customers SET tg_chat_id=NULL WHERE tg_chat_id=?", (chat_id,))
        c.db.ex("UPDATE customers SET tg_chat_id=? WHERE id=?", (chat_id, cid))
        name = c.db.scalar("SELECT full_name FROM customers WHERE id=?", (cid,))
        await cb.answer()
        if cb.message:
            await cb.message.answer(f"Тепер ви — {name} (клієнт #{cid}). Опишіть проблему текстом, фото чи відео.")

    async def respond(message: Message, text: str, media=None) -> None:  # noqa: ANN001
        cid = customer_for(message.chat.id)
        await message.bot.send_chat_action(message.chat.id, "typing")
        reply = await c.pipeline.handle(cid, text, media)
        await message.answer(reply.text)

    @router.message(F.text & ~F.text.startswith("/"))
    async def on_text(message: Message) -> None:
        await respond(message, message.text or "")

    @router.message(F.photo)
    async def on_photo(message: Message, bot: Bot) -> None:
        buf = io.BytesIO()
        await bot.download(message.photo[-1], destination=buf)
        fields = await c.media.from_images([buf.getvalue()], message.caption or "")
        buf.close()  # the original is dropped here; only extracted fields survive
        if fields is None and not message.caption:
            await message.answer("Не вдалося розпізнати зображення. Опишіть, будь ласка, проблему текстом.")
            return
        await respond(message, message.caption or "", fields)

    @router.message(F.video | F.video_note | F.animation)
    async def on_video(message: Message, bot: Bot) -> None:
        video = message.video or message.video_note or message.animation
        if video.file_size and video.file_size > MAX_VIDEO_BYTES:
            await message.answer("Відео більше 20 МБ. Надішліть, будь ласка, коротший запис або скріншот.")
            return
        buf = io.BytesIO()
        await bot.download(video, destination=buf)
        try:
            fields = await c.media.from_video(buf.getvalue(), message.caption or "")
        except MediaError as exc:
            log.warning("video processing failed: %s", exc)
            fields = None
        finally:
            buf.close()
        if fields is None and not message.caption:
            await message.answer("Не вдалося обробити відео. Надішліть скріншот або опишіть проблему текстом.")
            return
        await respond(message, message.caption or "", fields)

    @router.message(F.document)
    async def on_document(message: Message, bot: Bot) -> None:
        doc = message.document
        if not (doc.mime_type or "").startswith("image/") or (doc.file_size or 0) > MAX_VIDEO_BYTES:
            await message.answer("Я приймаю текст, фото або відео до 20 МБ.")
            return
        buf = io.BytesIO()
        await bot.download(doc, destination=buf)
        fields = await c.media.from_images([buf.getvalue()], message.caption or "")
        buf.close()
        await respond(message, message.caption or "", fields)

    return router
