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
from app.container import Container
from app.db.repo import money

log = logging.getLogger(__name__)


def build_operator_router(c: Container, notifier: TelegramNotifier) -> Router:
    router = Router(name="operators")
    op_chat = c.settings.operator_chat_id
    if op_chat:
        router.message.filter(F.chat.id == op_chat)
        router.callback_query.filter(F.message.chat.id == op_chat)

    def who(cb: CallbackQuery) -> str:
        u = cb.from_user
        return f"{u.full_name} (@{u.username})" if u.username else f"{u.full_name} ({u.id})"

    @router.message(Command("help", "start"))
    async def help_(message: Message) -> None:
        log.info("/help from chat %s (%s)", message.chat.id, message.chat.type)
        await message.answer("Команди: /incidents, /stats, /simulate " + simulate.SCENARIO_HELP +
                             f"\nOPERATOR_CHAT_ID цього чату: <code>{message.chat.id}</code>", parse_mode="HTML")

    @router.message(Command("stats"))
    async def stats_(message: Message) -> None:
        await message.answer(stats.render(stats.compute(c.db)), parse_mode="HTML")

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
            await cb.answer("Виконую план…")
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
            await cb.answer("План відхилено" if ok else "Вже оброблено")
        if incident_id:
            await notifier.incident_changed(int(incident_id))

    @router.callback_query(F.data.startswith("inc:close:"))
    async def close_cb(cb: CallbackQuery) -> None:
        incident_id = int(cb.data.split(":")[2])
        result = await c.incidents.close(incident_id, who(cb))
        await cb.answer(result.message, show_alert=not result.ok)
        if result.ok:
            await cb.message.answer(f"🔒 {html.escape(result.message)}")

    @router.callback_query(F.data.startswith("esc:"))
    async def esc_cb(cb: CallbackQuery) -> None:
        _, verb, esc_id = cb.data.split(":")
        msg = await c.executor.decide_escalation(int(esc_id), verb in ("ok", "take"), who(cb))
        await cb.answer(msg[:190], show_alert=True)
        await notifier.refresh_escalation(int(esc_id))

    return router
