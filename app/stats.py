"""Support KPIs for /stats. Minutes saved is an ESTIMATE (constants below, see README)."""
from __future__ import annotations

from dataclasses import dataclass

from app.db.repo import Database, money

# Estimated operator minutes saved per event.
MIN_PER_AUTO_RESOLVED = 6     # ticket fully handled by the agent
MIN_PER_INSTANT_LINK = 8      # ticket answered with incident status instead of a fresh investigation
MIN_PER_PROACTIVE = 10        # affected customer helped before they had to write (avoided ticket)


@dataclass
class Stats:
    tickets: int
    auto_resolved: int
    escalated: int
    instant_linked: int
    proactive_helped: int
    at_risk_protected: int
    fraud_blocked: int
    refunded_total: float
    minutes_saved: int


def compute(db: Database) -> Stats:
    s = lambda sql, p=(): db.scalar(sql, p) or 0  # noqa: E731
    tickets = s("SELECT COUNT(*) FROM tickets")
    auto = s("SELECT COUNT(*) FROM tickets WHERE status IN ('auto_resolved','linked','resolved') "
             "AND id NOT IN (SELECT ticket_id FROM escalations WHERE ticket_id IS NOT NULL AND status != 'closed')")
    escalated = s("SELECT COUNT(DISTINCT ticket_id) FROM escalations WHERE status != 'closed'")
    linked = s("SELECT COUNT(*) FROM tickets WHERE link_method IN ('signature','vector','category')")
    proactive = s(
        "SELECT COUNT(DISTINCT a.customer_id) FROM actions a JOIN incident_members m "
        "ON m.incident_id=a.incident_id AND m.customer_id=a.customer_id AND m.role='affected' "
        "WHERE a.actor='operator' AND NOT EXISTS (SELECT 1 FROM tickets t WHERE t.customer_id=a.customer_id "
        "AND t.incident_id=a.incident_id)")
    at_risk = s(
        "SELECT COUNT(*) FROM incident_members m WHERE role='at_risk' AND NOT EXISTS (SELECT 1 FROM actions a "
        "WHERE a.customer_id=m.customer_id AND a.incident_id=m.incident_id AND a.action_type='notify_customer')")
    fraud = s(
        "SELECT COUNT(*) FROM actions a LEFT JOIN tickets t ON t.id=a.ticket_id LEFT JOIN incidents i "
        "ON i.id=a.incident_id WHERE a.action_type='block_card' AND (t.category IN ('social_engineering','phishing') "
        "OR i.category='phishing')")
    refunded = float(s("SELECT SUM(amount) FROM actions WHERE action_type='refund_transaction'"))
    minutes = auto * MIN_PER_AUTO_RESOLVED + linked * MIN_PER_INSTANT_LINK + proactive * MIN_PER_PROACTIVE
    return Stats(tickets, auto, escalated, linked, proactive, at_risk, fraud, round(refunded, 2), minutes)


def render(st: Stats) -> str:
    return (
        "📊 <b>Статистика підтримки</b>\n"
        f"Звернень усього: {st.tickets}\n"
        f"🤖 Вирішено автоматично: <b>{st.auto_resolved}</b>\n"
        f"🙋 Ескальовано оператору: {st.escalated}\n"
        f"🔗 Миттєво привʼязано до відомого інциденту: <b>{st.instant_linked}</b>\n"
        f"📤 Affected, яким допомогли без їхнього звернення: <b>{st.proactive_helped}</b>\n"
        f"🛡 At_risk, захищених від зайвих повідомлень: {st.at_risk_protected}\n"
        f"🔒 Заблоковано шахрайських спроб (блок карток): {st.fraud_blocked}\n"
        f"💸 Повернено коштів (після підтвердження): {money(st.refunded_total)} грн\n"
        f"⏱ Оцінка зекономленого часу операторів: ~<b>{st.minutes_saved} хв</b> "
        f"(авто×{MIN_PER_AUTO_RESOLVED} + привʼязка×{MIN_PER_INSTANT_LINK} + проактивно×{MIN_PER_PROACTIVE})"
    )


def render_dashboard(db: Database) -> str:
    """Decision distribution and top escalation reasons."""
    total = db.scalar("SELECT COUNT(*) FROM decisions") or 0
    dist = {r["final_decision"]: r["n"] for r in db.q(
        "SELECT final_decision, COUNT(*) AS n FROM decisions GROUP BY final_decision")}
    security = db.scalar("SELECT COUNT(*) FROM decisions WHERE security_review=1") or 0
    strengthened = db.scalar("SELECT COUNT(*) FROM decisions WHERE llm_decision='auto_approve' "
                             "AND final_decision!='auto_approve'") or 0
    auto_sum = db.scalar("SELECT SUM(amount) FROM actions WHERE actor='agent' "
                         "AND action_type IN ('refund_transaction','refund_credit_charge')") or 0
    reasons: dict[str, int] = {}
    import json
    for r in db.q("SELECT enforced_json, factors_json FROM decisions WHERE final_decision='needs_operator'"):
        for item in json.loads(r["enforced_json"] or "[]"):
            key = item.split("(")[0].split(":")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
    for r in db.q("SELECT t.category, COUNT(*) AS n FROM escalations e JOIN tickets t ON t.id=e.ticket_id "
                  "WHERE e.status != 'closed' GROUP BY t.category"):
        reasons[f"ескалація: {r['category']}"] = reasons.get(f"ескалація: {r['category']}", 0) + r["n"]
    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:6]
    lines = [
        "📈 <b>Дашборд рішень</b>",
        f"Рішень про гроші: {total}",
        f"✅ auto_approve: <b>{dist.get('auto_approve', 0)}</b> (повернено автоматично {money(float(auto_sum))} грн)",
        f"👩‍💼 needs_operator: <b>{dist.get('needs_operator', 0)}</b>",
        f"🔐 security_review: <b>{security}</b>",
        f"ℹ️ info_only: {dist.get('info_only', 0)}",
        f"🛡 policy посилила рішення LLM (auto → operator): <b>{strengthened}</b>",
        "",
        "<b>Топ причин ескалації:</b>",
    ]
    lines += [f"  {n}× {reason}" for reason, n in top] or ["  —"]
    return "\n".join(lines)
