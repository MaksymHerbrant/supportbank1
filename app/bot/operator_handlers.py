"""Operator group: inline approvals, /stats, /simulate, /incidents."""
from __future__ import annotations

import html
import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, FSInputFile, Message

from app import stats
from app.bot import simulate
from app.bot.cards import TelegramNotifier
from app.bot.live_chat import LiveChats
from app.bot.safe import answer_cb
from app.container import Container
from app.db.repo import money, now_ts

log = logging.getLogger(__name__)


def build_operator_router(c: Container, notifier: TelegramNotifier) -> Router:
    router = Router(name="operators")
    op_chat = c.settings.operator_chat_id
    if op_chat:
        router.message.filter(F.chat.id == op_chat)
        router.callback_query.filter(F.message.chat.id == op_chat)

    live = LiveChats(c.db)

    async def open_chat(cid: int, operator: str, reply_to: Message) -> None:
        chat, created = live.open(cid, operator)
        name = c.db.scalar("SELECT full_name FROM customers WHERE id=?", (cid,))
        has_tg = bool(c.db.scalar("SELECT tg_chat_id FROM customers WHERE id=?", (cid,)))
        if created:
            await c.tools.notifier.send_customer(
                cid, f"👩‍💼 До чату підключився оператор {operator.split(' (')[0]}. Тепер вам відповідає людина.")
        history = c.pipeline.history(cid, limit=4) or "—"
        msg = await reply_to.answer(
            f"💬 <b>Живий чат з клієнтом #{cid} {html.escape(name or '')}</b> — веде {html.escape(chat['operator_name'])}\n"
            f"Відповідайте <b>реплаєм</b> на це або будь-яке повідомлення клієнта. /end (реплаєм) — завершити.\n"
            + ("" if has_tg else "⚠️ Демо-персонаж без Telegram: повідомлення будуть лише збережені.\n")
            + f"\n<i>Останні повідомлення:</i>\n{html.escape(history)}", parse_mode="HTML")
        live.remember(msg.message_id, cid, "card")

    def who(cb: CallbackQuery) -> str:
        u = cb.from_user
        return f"{u.full_name} (@{u.username})" if u.username else f"{u.full_name} ({u.id})"

    @router.message(Command("help", "start"))
    async def help_(message: Message) -> None:
        log.info("/help from chat %s (%s)", message.chat.id, message.chat.type)
        await message.answer("Команди: /incidents, /stats, /dashboard, /top, /chat <id>, /end, /simulate " + simulate.SCENARIO_HELP +
                             f"\nOPERATOR_CHAT_ID цього чату: <code>{message.chat.id}</code>", parse_mode="HTML")

    @router.message(Command("stats"))
    async def stats_(message: Message) -> None:
        await message.answer(stats.render(stats.compute(c.db)), parse_mode="HTML")

    @router.message(Command("dashboard"))
    async def dashboard_(message: Message) -> None:
        day = now_ts()[:10] + " 00:00:00"
        await message.answer(stats.render_dashboard(c.db) + "\n\n" + stats.render_top(c.db, day), parse_mode="HTML")

    @router.message(Command("top"))
    async def top_(message: Message) -> None:
        await message.answer(stats.render_top(c.db, now_ts()[:10] + " 00:00:00"), parse_mode="HTML")

    @router.message(Command("incidents"))
    async def incidents_(message: Message) -> None:
        rows = c.db.q(
            "SELECT i.*, (SELECT COUNT(*) FROM incident_members m WHERE m.incident_id=i.id AND role='affected') aff, "
            "(SELECT COUNT(*) FROM incident_members m WHERE m.incident_id=i.id AND role='at_risk') risk, "
            "(SELECT COUNT(*) FROM tickets t WHERE t.incident_id=i.id) linked "
            "FROM incidents i WHERE status='open' ORDER BY id")
        if not rows:
            await message.answer("Відкритих інцидентів немає ✅")
            return
        lines = ["📂 <b>Відкриті інциденти</b>"]
        for r in rows:
            lines.append(f"#{r['id']} [{r['severity']}] {html.escape(r['summary'])}\n"
                         f"   affected {r['aff']} · at_risk {r['risk']} · звернень {r['linked']}")
        await message.answer("\n".join(lines), parse_mode="HTML")
        for r in rows:
            await notifier.incident_changed(r["id"])

    @router.message(Command("simulate"))
    async def simulate_(message: Message, command: CommandObject) -> None:
        name = (command.args or "").strip()
        if not name:
            await message.answer("Використання: /simulate " + simulate.SCENARIO_HELP)
            return

        async def say(text: str) -> None:
            await message.answer(text, parse_mode="HTML")

        async def send_photo(path, caption: str) -> None:  # noqa: ANN001
            await message.answer_photo(FSInputFile(path), caption=caption)

        await simulate.run(c, name, say, send_photo)

    @router.callback_query(F.data.startswith("plan:"))
    async def plan_cb(cb: CallbackQuery) -> None:
        _, verb, plan_id = cb.data.split(":")
        incident_id = c.db.scalar("SELECT incident_id FROM plans WHERE id=?", (int(plan_id),))
        if verb == "ok":
            await answer_cb(cb, "Виконую план…")
            report = await c.executor.approve_plan(int(plan_id), who(cb))
            if report.already_processed:
                await cb.message.answer(f"ℹ️ План #{plan_id} вже оброблено (статус: {report.status}). "
                                        f"Повторне натискання нічого не дублює.")
            else:
                text = (f"✅ План #{plan_id} підтвердив {html.escape(who(cb))}: виконано {report.done} дій"
                        + (f", помилок {report.failed}" if report.failed else "")
                        + (f", повернено {money(report.refunded_total)} грн" if report.refunded_total else "")
                        + f", повідомлено affected-клієнтів: {report.notified}.")
                if report.sample_message:
                    text += f"\n📤 Приклад повідомлення: «{html.escape(report.sample_message)}»"
                await cb.message.answer(text, parse_mode="HTML")
        else:
            ok = c.executor.reject_plan(int(plan_id), who(cb))
            await answer_cb(cb, "План відхилено" if ok else "Вже оброблено")
        if incident_id:
            await notifier.incident_changed(int(incident_id))

    @router.callback_query(F.data.startswith("inc:close:"))
    async def close_cb(cb: CallbackQuery) -> None:
        incident_id = int(cb.data.split(":")[2])
        result = await c.incidents.close(incident_id, who(cb))
        await answer_cb(cb, result.message, show_alert=not result.ok)
        if result.ok:
            await cb.message.answer(f"🔒 {html.escape(result.message)}")

    @router.callback_query(F.data.startswith("esc:chat:"))
    async def esc_chat_cb(cb: CallbackQuery) -> None:
        cid = c.db.scalar("SELECT customer_id FROM escalations WHERE id=?", (int(cb.data.split(":")[2]),))
        if cid is None:
            await answer_cb(cb, "Клієнта не знайдено", show_alert=True)
            return
        await answer_cb(cb, "Чат відкрито")
        await open_chat(int(cid), who(cb), cb.message)

    @router.message(Command("chat"))
    async def chat_cmd(message: Message, command: CommandObject) -> None:
        arg = (command.args or "").strip().lstrip("#")
        cid = int(arg) if arg.isdigit() else (
            live.customer_of(message.reply_to_message.message_id) if message.reply_to_message else None)
        if cid is None:
            await message.answer("Використання: /chat <id клієнта> або реплай /chat на повідомлення клієнта")
            return
        await open_chat(cid, message.from_user.full_name, message)

    @router.message(Command("end"))
    async def end_cmd(message: Message, command: CommandObject) -> None:
        arg = (command.args or "").strip().lstrip("#")
        cid = int(arg) if arg.isdigit() else (
            live.customer_of(message.reply_to_message.message_id) if message.reply_to_message else None)
        if cid is None or not live.close(cid):
            await message.answer("Відкритого чату не знайдено. Використання: /end <id> або реплай /end")
            return
        await c.tools.notifier.send_customer(cid, "Оператор завершив чат. Якщо з'являться питання — пишіть, "
                                                  "вам знову допоможе AI-асистент.")
        await message.answer(f"✅ Чат з клієнтом #{cid} завершено, далі відповідає AI.")

    @router.message(F.reply_to_message, F.text, ~F.text.startswith("/"))
    async def operator_reply(message: Message) -> None:
        cid = live.customer_of(message.reply_to_message.message_id)
        if cid is None:
            return
        operator = message.from_user.full_name
        if live.active(cid) is None:
            await open_chat(cid, operator, message)
        delivered = await c.tools.notifier.send_customer(cid, f"👩‍💼 {operator}: {message.text}")
        live.remember(message.message_id, cid, "to_customer", message.text)
        if not delivered:
            await message.reply("ℹ️ Клієнт — демо-персонаж без Telegram: повідомлення збережено, але не доставлено.")

    @router.callback_query(F.data.startswith("esc:"))
    async def esc_cb(cb: CallbackQuery) -> None:
        _, verb, esc_id = cb.data.split(":")
        msg = await c.executor.decide_escalation(int(esc_id), verb in ("ok", "take"), who(cb))
        await answer_cb(cb, msg[:190], show_alert=True)
        await notifier.refresh_escalation(int(esc_id))

    return router
