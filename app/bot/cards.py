"""Operator-facing cards (incidents, escalations, bug tickets) and the Telegram notifier."""
from __future__ import annotations

import html
import logging
from datetime import datetime, timezone

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from app.db.repo import Database, loads, money, parse_ts
from app.incidents.linker import linked_counts
from app.incidents.planner import Planner

log = logging.getLogger(__name__)
SEV_ICON = {"critical": "🟥", "high": "🟧", "medium": "🟨", "low": "🟩"}
LEVEL_NAME = {1: "FAQ", 2: "безпечна дія", 3: "розслідування", 4: "тільки людина"}


def esc(text: object) -> str:
    return html.escape(str(text if text is not None else ""))


def _age(created_at: str) -> str:
    minutes = int((datetime.now(timezone.utc) - parse_ts(created_at)).total_seconds() // 60)
    return f"{minutes} хв" if minutes < 120 else f"{minutes // 60} год {minutes % 60} хв"


def render_incident(db: Database, incident_id: int) -> tuple[str, InlineKeyboardMarkup | None]:
    inc = db.q1("SELECT * FROM incidents WHERE id=?", (incident_id,))
    if inc is None:
        return "Інцидент не знайдено", None
    roles = {r["role"]: r for r in db.q(
        "SELECT role, COUNT(*) AS n, SUM(amount) AS amt, SUM(promoted_at IS NOT NULL) AS promoted "
        "FROM incident_members WHERE incident_id=? GROUP BY role", (incident_id,))}
    aff = roles.get("affected")
    risk_n = roles["at_risk"]["n"] if "at_risk" in roles else 0
    links = linked_counts(db, incident_id)
    total_links = sum(links.values())
    status = "🟢 ВИРІШЕНО" if inc["status"] == "resolved" else f"відкритий {_age(inc['created_at'])}"
    lines = [
        f"🚨 <b>Інцидент #{inc['id']}</b> · {SEV_ICON.get(inc['severity'], '')} {inc['severity'].upper()} · {status}",
        f"<b>{esc(inc['summary'])}</b>",
        "",
        f"🧠 <b>Гіпотеза:</b> {esc(inc['hypothesis'] or '—')}",
    ]
    evidence = loads(inc["evidence_json"], [])
    if evidence:
        lines.append("📎 <b>Докази:</b>")
        lines += [f"  • {esc(e)}" for e in evidence[:4]]
    promoted = int(aff["promoted"] or 0) if aff else 0
    lines += [
        "",
        f"👥 <b>Affected (підтверджено даними): {aff['n'] if aff else 0}</b>"
        + (f" · з них {promoted} перейшли з at_risk" if promoted else ""),
        f"🛡 <b>at_risk: {risk_n} клієнтів</b> — без повідомлень, враховано в пріоритеті тікета",
        f"🔗 Привʼязано звернень: <b>{total_links}</b> ("
        + " · ".join(f"{k} {v}" for k, v in sorted(links.items())) + ")" if total_links else "🔗 Привʼязано звернень: 0",
    ]
    planner = Planner(db)
    pending = db.q1("SELECT id FROM plans WHERE incident_id=? AND status='pending' ORDER BY id DESC LIMIT 1",
                    (incident_id,))
    done = db.q1("SELECT COUNT(*) AS n, SUM(CASE WHEN action_type='refund_transaction' THEN amount END) AS amt, "
                 "COUNT(DISTINCT customer_id) AS c FROM actions WHERE incident_id=? AND actor='operator'", (incident_id,))
    if done and done["n"]:
        lines.append(f"✅ Виконано дій: {done['n']} для {done['c']} клієнтів"
                     + (f", повернено {money(done['amt'])} грн" if done["amt"] else ""))
    buttons: list[list[InlineKeyboardButton]] = []
    if pending:
        pv = planner.preview(pending["id"])
        lines += ["", f"📋 <b>План #{pv.plan_id} — очікує підтвердження</b>: {pv.items} дій для {pv.customers} affected"
                  + (f", повернення {money(pv.refund_total)} грн" if pv.refund_total else "")]
        lines += [esc(line) for line in pv.lines]
        if pv.customers > len(pv.lines):
            lines.append(f"  … і ще {pv.customers - len(pv.lines)} клієнтів")
        buttons.append([
            InlineKeyboardButton(text="✅ Підтвердити", callback_data=f"plan:ok:{pv.plan_id}"),
            InlineKeyboardButton(text="❌ Відхилити", callback_data=f"plan:no:{pv.plan_id}"),
        ])
    if inc["status"] == "open":
        buttons.append([InlineKeyboardButton(text="🔒 Закрити інцидент", callback_data=f"inc:close:{incident_id}")])
    text = "\n".join(lines)
    return text[:4000], InlineKeyboardMarkup(inline_keyboard=buttons) if buttons else None


def render_escalation(db: Database, esc_id: int) -> tuple[str, InlineKeyboardMarkup | None]:
    e = db.q1("SELECT e.*, c.full_name, c.segment FROM escalations e LEFT JOIN customers c ON c.id=e.customer_id "
              "WHERE e.id=?", (esc_id,))
    if e is None:
        return "Ескалацію не знайдено", None
    t = db.q1("SELECT * FROM tickets WHERE id=?", (e["ticket_id"],)) if e["ticket_id"] else None
    lines = [f"🙋 <b>Ескалація #{e['id']}</b> · клієнт #{e['customer_id']} {esc(e['full_name'])}"
             + (" · VIP" if e["segment"] == "vip" else ""),
             f"Причина: {esc(e['reason'])}"]
    if t is not None:
        flags = loads(t["risk_flags_json"], [])
        lines += [f"Звернення: «{esc((t['text'] or '')[:400])}»",
                  f"Triage: {esc(t['category'])} · рівень {t['level']} ({LEVEL_NAME.get(t['level'], '')}) · "
                  f"conf {t['confidence']}" + (f" · ⚠️ {esc(', '.join(flags))}" if flags else "")]
    proposed = loads(e["proposed_json"], None) or {}
    lines += _risk_lines(db, e["customer_id"], proposed.get("decision_id"))
    buttons = None
    if e["status"] != "open":
        lines.append(f"Статус: {esc(e['status'])} ({esc(e['decided_by'] or '')})")
    elif proposed.get("action") in ("refund_transaction", "refund_credit_charge"):
        what = (f"повернення {money(proposed.get('amount'))} грн (tx #{proposed['tx_id']})"
                if proposed["action"] == "refund_transaction"
                else f"компенсація кредитного нарахування {money(proposed.get('amount'))} грн (#{proposed['charge_id']})")
        lines.append(f"💡 Пропозиція агента: {what} — потрібне підтвердження оператора")
        buttons = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="✅ Підтвердити", callback_data=f"esc:ok:{esc_id}"),
            InlineKeyboardButton(text="❌ Відхилити", callback_data=f"esc:no:{esc_id}")]])
    else:
        buttons = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="👤 Взяти в роботу", callback_data=f"esc:take:{esc_id}")]])
    return "\n".join(lines)[:4000], buttons


def _risk_lines(db: Database, customer_id: int | None, decision_id: int | None) -> list[str]:
    """Risk level, score, factors with evidence, LLM vs final decision and operator note."""
    if customer_id is None:
        return []
    from app.agent.risk import assess

    lines = []
    d = db.q1("SELECT * FROM decisions WHERE id=?", (decision_id,)) if decision_id else None
    if d is not None:
        level, score, factors = d["risk_level"], d["risk_score"], loads(d["factors_json"], [])
    else:
        _, ra = assess(db, customer_id)
        level, score, factors = ra.level, ra.score, [f.model_dump() for f in ra.factors]
    icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(level, "")
    lines.append(f"{icon} <b>Ризик: {level.upper()} ({score})</b>")
    lines += [f"  • +{f['points']} {esc(f['evidence'])}" for f in factors[:6]]
    if d is not None:
        if d["security_review"]:
            lines.append("🔐 <b>SECURITY REVIEW</b> — перевірте безпеку акаунта перед будь-якою дією")
        enforced = loads(d["enforced_json"], [])
        lines.append(f"🤖 LLM: {esc(d['llm_decision'] or '—')} → policy: <b>{esc(d['final_decision'])}</b>"
                     + (f" (посилено: {esc('; '.join(enforced))})" if enforced else ""))
        if d["operator_note"]:
            lines.append(f"📝 {esc(d['operator_note'])}")
    return lines


def render_bug(db: Database, bug_id: int) -> str:
    b = db.q1("SELECT * FROM bug_tickets WHERE id=?", (bug_id,))
    if b is None:
        return ""
    return (f"🐞 <b>BUG-{b['id']} [{b['priority']}]</b> {esc(b['title'])}\nSeverity: <b>{esc(b['severity'])}</b> · "
            f"Інцидент #{b['incident_id']}\n\n{esc(b['body'])}")


class TelegramNotifier:
    """Delivers to Telegram; fake customers without a chat get a logged 'simulated' delivery."""

    def __init__(self, bot: Bot, db: Database, operator_chat_id: int, dev_chat_id: int) -> None:
        self.bot = bot
        self.db = db
        self.operator_chat_id = operator_chat_id
        self.dev_chat_id = dev_chat_id

    async def send_customer(self, customer_id: int, text: str) -> bool:
        chat_id = self.db.scalar("SELECT tg_chat_id FROM customers WHERE id=?", (customer_id,))
        if not chat_id:
            log.info("simulated delivery to customer %s: %s", customer_id, text[:80])
            return False
        try:
            await self.bot.send_message(chat_id, text)
            return True
        except TelegramAPIError as exc:
            log.warning("send to customer %s failed: %s", customer_id, exc)
            return False

    async def incident_changed(self, incident_id: int) -> None:
        if not self.operator_chat_id:
            return
        text, kb = render_incident(self.db, incident_id)
        msg_id = self.db.scalar("SELECT operator_msg_id FROM incidents WHERE id=?", (incident_id,))
        if msg_id:
            try:
                await self.bot.edit_message_text(text, chat_id=self.operator_chat_id, message_id=msg_id,
                                                 reply_markup=kb, parse_mode="HTML")
                return
            except TelegramBadRequest as exc:
                if "not modified" in str(exc):
                    return
                log.warning("edit incident card failed (%s); reposting", exc)
        try:
            msg = await self.bot.send_message(self.operator_chat_id, text, reply_markup=kb, parse_mode="HTML")
            self.db.ex("UPDATE incidents SET operator_msg_id=? WHERE id=?", (msg.message_id, incident_id))
        except TelegramAPIError as exc:
            log.error("posting incident card failed: %s", exc)

    async def escalation_created(self, escalation_id: int) -> None:
        if not self.operator_chat_id:
            return
        text, kb = render_escalation(self.db, escalation_id)
        try:
            msg = await self.bot.send_message(self.operator_chat_id, text, reply_markup=kb, parse_mode="HTML")
            self.db.ex("UPDATE escalations SET operator_msg_id=? WHERE id=?", (msg.message_id, escalation_id))
        except TelegramAPIError as exc:
            log.error("posting escalation failed: %s", exc)

    async def dev_ticket(self, bug_ticket_id: int) -> None:
        chat = self.dev_chat_id or self.operator_chat_id
        if not chat:
            return
        try:
            await self.bot.send_message(chat, render_bug(self.db, bug_ticket_id), parse_mode="HTML")
        except TelegramAPIError as exc:
            log.error("posting bug ticket failed: %s", exc)

    async def refresh_escalation(self, escalation_id: int) -> None:
        msg_id = self.db.scalar("SELECT operator_msg_id FROM escalations WHERE id=?", (escalation_id,))
        if not msg_id or not self.operator_chat_id:
            return
        text, kb = render_escalation(self.db, escalation_id)
        try:
            await self.bot.edit_message_text(text, chat_id=self.operator_chat_id, message_id=msg_id,
                                             reply_markup=kb, parse_mode="HTML")
        except TelegramBadRequest:
            pass
