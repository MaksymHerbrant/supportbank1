"""Customer bot: text, photos, videos. Media is processed in memory and never stored."""
from __future__ import annotations

import io
import logging

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.agent.media import MAX_VIDEO_BYTES, MediaError
from app.agent import policy
from app.bot import personas
from app.bot.live_chat import LiveChats
from app.container import Container

log = logging.getLogger(__name__)

def build_client_router(c: Container) -> Router:
    router = Router(name="clients")
    router.message.filter(F.chat.type == "private")
    live = LiveChats(c.db)
    op_chat = c.settings.operator_chat_id

    async def relay(message: Message) -> bool:
        """If an operator is connected, forward the message to the operator group instead of the AI."""
        cid = c.db.scalar("SELECT id FROM customers WHERE tg_chat_id=?", (message.chat.id,))
        chat = live.active(int(cid)) if cid else None
        if not chat or not op_chat:
            return False
        name = c.db.scalar("SELECT full_name FROM customers WHERE id=?", (cid,))
        header = f"👤 #{cid} {name} → {chat['operator_name']}"
        if message.text:
            text = policy.mask_pan(message.text) or ""
            sent = await message.bot.send_message(op_chat, f"{header}:\n{text}")
        else:
            text = policy.mask_pan(message.caption or "") or ""
            sent = await message.copy_to(op_chat, caption=f"{header}: {text}"[:1000])
        live.remember(sent.message_id, int(cid), "to_operator", text or "(медіа)")
        return True

    def persona_keyboard() -> InlineKeyboardMarkup:
        rows = [[InlineKeyboardButton(text=f"{name} (#{cid})", callback_data=f"persona:{cid}")]
                for _, cid, name in personas.listing(c.db)]
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def customer_for(chat_id: int) -> int:
        cid = c.db.scalar("SELECT id FROM customers WHERE tg_chat_id=?", (chat_id,))
        if cid:
            return int(cid)
        cid = personas.customer_by_slug(c.db, personas.DEFAULT_PERSONA)
        personas.bind(c.db, chat_id, cid)
        return int(cid)

    @router.message(CommandStart())
    @router.message(Command("persona"))
    async def start(message: Message) -> None:
        await message.answer(
            "Вітаю! Я — AI-асистент підтримки DemoBank (демо, усі дані вигадані).\n"
            "Оберіть клієнта, від імені якого будете писати (або /demo_login &lt;persona&gt;). "
            "Проблему не обирайте — просто опишіть її, бот розбереться сам.",
            reply_markup=persona_keyboard(), parse_mode="HTML")

    @router.message(Command("demo_login"))
    async def demo_login(message: Message, command: CommandObject) -> None:
        cid = personas.customer_by_slug(c.db, command.args or "")
        if cid is None:
            await message.answer("Використання: /demo_login " + " | ".join(personas.PERSONAS))
            return
        name = personas.bind(c.db, message.chat.id, cid)
        await message.answer(f"Тепер ви — {name} (клієнт #{cid}). Опишіть, що сталося.")

    @router.callback_query(F.data.startswith("persona:"))
    async def choose(cb: CallbackQuery) -> None:
        cid = int(cb.data.split(":")[1])
        chat_id = cb.message.chat.id if cb.message else cb.from_user.id
        name = personas.bind(c.db, chat_id, cid)
        await cb.answer()
        if cb.message:
            await cb.message.answer(f"Тепер ви — {name} (клієнт #{cid}). Опишіть, що сталося: текстом, фото чи відео.")

    async def respond(message: Message, text: str, media=None) -> None:  # noqa: ANN001
        cid = customer_for(message.chat.id)
        await message.bot.send_chat_action(message.chat.id, "typing")
        reply = await c.pipeline.handle(cid, text, media)
        await message.answer(reply.text)

    @router.message(F.text & ~F.text.startswith("/"))
    async def on_text(message: Message) -> None:
        if await relay(message):
            return
        await respond(message, message.text or "")

    @router.message(F.photo)
    async def on_photo(message: Message, bot: Bot) -> None:
        if await relay(message):
            return
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
        if await relay(message):
            return
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
        if await relay(message):
            return
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
