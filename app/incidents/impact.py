"""Deterministic impact analysis. The ONLY place that decides who is affected.

affected  = the problem already happened to the customer and the data proves it
            (transactions, app logs, ATM events). Only they may get messages/actions.
at_risk   = the problem may happen (upcoming recurring payment, same app version...).
            Used for severity, bug priority and the operator card only. Never contacted.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

from app.agent.triage import Signature
from app.db.repo import Database, ago, dumps, local_time, now_ts, parse_ts, ts, utcnow

log = logging.getLogger(__name__)
DUP_WINDOW_S = 120
LOOKBACK_HOURS = {"double_charge": 24, "phishing": 72, "atm_capture": 48, "app_payment_error": 24}
AT_RISK_DAYS = 7


@dataclass
class Member:
    customer_id: int
    reason: str
    evidence: list[str] = field(default_factory=list)
    amount: float | None = None
    ref_id: str | None = None      # comma-separated ids (tx / event) backing the claim
    card_id: int | None = None


@dataclass
class Impact:
    affected: list[Member]
    at_risk: list[Member]


@dataclass
class SyncResult:
    new_affected: list[int]
    promoted: list[int]           # moved from at_risk to affected
    affected_count: int
    at_risk_count: int


def _since(created_at: str | None, category: str) -> str:
    hours = LOOKBACK_HOURS.get(category, 24)
    if created_at:
        return ts(parse_ts(created_at) - timedelta(hours=hours))
    return ago(hours=hours)


def duplicate_pairs(db: Database, merchant_id: str | None, since: str, customer_id: int | None = None) -> list[dict]:
    """Same card, merchant and amount within DUP_WINDOW_S seconds."""
    sql = (
        "SELECT a.customer_id, a.card_id, a.id AS first_id, b.id AS dup_id, a.amount, a.merchant_id, "
        "a.created_at AS first_at, b.created_at AS dup_at, b.status AS dup_status, "
        "CAST(strftime('%s', b.created_at) - strftime('%s', a.created_at) AS INTEGER) AS gap_s "
        "FROM transactions a JOIN transactions b ON b.customer_id=a.customer_id AND b.card_id=a.card_id "
        "AND b.merchant_id=a.merchant_id AND b.amount=a.amount AND b.id > a.id "
        "AND strftime('%s', b.created_at) - strftime('%s', a.created_at) BETWEEN 0 AND ? "
        "WHERE a.created_at >= ? AND a.status != 'failed' AND b.status != 'failed'"
    )
    params: list = [DUP_WINDOW_S, since]
    if merchant_id:
        sql += " AND a.merchant_id=?"
        params.append(merchant_id)
    if customer_id is not None:
        sql += " AND a.customer_id=?"
        params.append(customer_id)
    return [dict(r) for r in db.q(sql + " ORDER BY a.created_at", params)]


def _double_charge(db: Database, sig: Signature, since: str) -> Impact:
    by_customer: dict[int, Member] = {}
    for p in duplicate_pairs(db, sig.merchant_id, since):
        m = by_customer.setdefault(p["customer_id"], Member(
            p["customer_id"], reason="", amount=0.0, card_id=p["card_id"], ref_id=""))
        m.amount = round((m.amount or 0) + p["amount"], 2)
        m.ref_id = ",".join(filter(None, [m.ref_id, str(p["dup_id"])]))
        m.evidence.append(f"tx #{p['first_id']} і #{p['dup_id']}: {p['amount']:.2f} грн, інтервал {p['gap_s']} с, "
                          f"{local_time(p['first_at'])}")
    for m in by_customer.values():
        m.reason = f"дубль транзакції в даних ({len(m.evidence)} пар(и), {m.amount:.2f} грн)"
    affected = list(by_customer.values())
    today = utcnow().strftime("%Y-%m-%d")
    horizon = (utcnow() + timedelta(days=AT_RISK_DAYS)).strftime("%Y-%m-%d")
    at_risk = [
        Member(r["customer_id"], reason=f"регулярний платіж {r['amount']:.0f} грн {r['next_date']}",
               evidence=[f"recurring #{r['id']}"], amount=r["amount"], ref_id=str(r["id"]), card_id=r["card_id"])
        for r in db.q("SELECT * FROM recurring_payments WHERE merchant_id=? AND active=1 AND next_date BETWEEN ? AND ?",
                      (sig.merchant_id, today, horizon))
        if r["customer_id"] not in by_customer
    ]
    return Impact(affected, _dedupe(at_risk))


def _phishing(db: Database, sig: Signature, since: str) -> Impact:
    rows = db.q(
        "SELECT t.id, t.customer_id, t.card_id, t.amount, t.created_at FROM transactions t "
        "JOIN merchants m ON m.id=t.merchant_id WHERE m.is_scam=1 AND m.scam_domain=? AND t.is_fraud=1 "
        "AND t.created_at >= ?", (sig.scam_domain, since))
    by_customer: dict[int, Member] = {}
    for r in rows:
        m = by_customer.setdefault(r["customer_id"], Member(r["customer_id"], "", amount=0.0, card_id=r["card_id"], ref_id=""))
        m.amount = round((m.amount or 0) + r["amount"], 2)
        m.ref_id = ",".join(filter(None, [m.ref_id, str(r["id"])]))
        m.evidence.append(f"tx #{r['id']}: {r['amount']:.2f} грн → шахрайський мерчант ({sig.scam_domain}), "
                          f"{local_time(r['created_at'])}")
    for m in by_customer.values():
        m.reason = f"підтверджена шахрайська транзакція після переходу на {sig.scam_domain}"
    at_risk = [
        Member(r["customer_id"], reason=f"перехід на {sig.scam_domain} без списань",
               evidence=[f"app_event #{r['id']}"], ref_id=str(r["id"]))
        for r in db.q("SELECT id, customer_id FROM app_events WHERE event_type='phishing_link_open' "
                      "AND details LIKE ? AND created_at >= ?", (f"%{sig.scam_domain}%", since))
        if r["customer_id"] not in by_customer
    ]
    return Impact(list(by_customer.values()), _dedupe(at_risk))


def _atm(db: Database, sig: Signature, since: str) -> Impact:
    affected: dict[int, Member] = {}
    for r in db.q("SELECT * FROM atm_events WHERE atm_id=? AND event_type='card_captured' AND created_at >= ?",
                  (sig.atm_id, since)):
        affected.setdefault(r["customer_id"], Member(
            r["customer_id"], reason=f"картку захоплено банкоматом {sig.atm_id}",
            evidence=[f"atm_event #{r['id']}: card_captured ({r['details']}), {local_time(r['created_at'])}"],
            ref_id=str(r["id"]), card_id=r["card_id"]))
    week = ago(days=AT_RISK_DAYS)
    at_risk = [
        Member(r["customer_id"], reason=f"користувався {sig.atm_id} за останні {AT_RISK_DAYS} днів",
               evidence=[f"atm_event #{r['id']}"], ref_id=str(r["id"]))
        for r in db.q("SELECT id, customer_id FROM atm_events WHERE atm_id=? AND event_type='withdrawal' "
                      "AND created_at >= ?", (sig.atm_id, week))
        if r["customer_id"] not in affected
    ]
    return Impact(list(affected.values()), _dedupe(at_risk))


def _app(db: Database, sig: Signature, since: str) -> Impact:
    sql = "SELECT * FROM app_events WHERE app_version=? AND event_type='payment_failed' AND created_at >= ?"
    params: list = [sig.app_version, since]
    if sig.error_code:
        sql += " AND error_code=?"
        params.append(sig.error_code)
    affected: dict[int, Member] = {}
    for r in db.q(sql, params):
        m = affected.setdefault(r["customer_id"], Member(
            r["customer_id"], reason=f"невдалі оплати у версії {sig.app_version}", ref_id=""))
        m.ref_id = ",".join(filter(None, [m.ref_id, str(r["id"])]))
        m.evidence.append(f"app_event #{r['id']}: payment_failed {r['error_code']} v{r['app_version']} "
                          f"«{r['details']}», {local_time(r['created_at'])}")
    week = ago(days=AT_RISK_DAYS)
    at_risk = [
        Member(r["customer_id"], reason=f"використовує версію {sig.app_version} без помилок",
               evidence=[f"app_event #{r['id']}"], ref_id=str(r["id"]))
        for r in db.q("SELECT id, customer_id FROM app_events WHERE app_version=? AND event_type='app_open' "
                      "AND created_at >= ?", (sig.app_version, week))
        if r["customer_id"] not in affected
    ]
    return Impact(list(affected.values()), _dedupe(at_risk))


def _dedupe(members: list[Member]) -> list[Member]:
    seen: dict[int, Member] = {}
    for m in members:
        seen.setdefault(m.customer_id, m)
    return list(seen.values())


CALCULATORS = {"double_charge": _double_charge, "phishing": _phishing, "atm_capture": _atm, "app_payment_error": _app}


def compute(db: Database, sig: Signature, created_at: str | None = None) -> Impact:
    """Compute affected / at_risk sets for a signature with SQL only."""
    calc = CALCULATORS.get(sig.category)
    if calc is None or not sig.key():
        return Impact([], [])
    return calc(db, sig, _since(created_at, sig.category))


def sync_members(db: Database, incident_id: int) -> SyncResult:
    """Recompute impact and persist it. Affected never goes back to at_risk; at_risk -> affected is recorded."""
    inc = db.q1("SELECT signature_json, created_at FROM incidents WHERE id=?", (incident_id,))
    if inc is None:
        return SyncResult([], [], 0, 0)
    sig = Signature.model_validate_json(inc["signature_json"])
    impact = compute(db, sig, inc["created_at"])
    existing = {r["customer_id"]: r["role"] for r in
                db.q("SELECT customer_id, role FROM incident_members WHERE incident_id=?", (incident_id,))}
    new_affected, promoted = [], []
    now = now_ts()
    for m in impact.affected:
        role = existing.get(m.customer_id)
        if role == "affected":
            db.ex("UPDATE incident_members SET evidence_json=?, amount=?, ref_id=?, reason=? "
                  "WHERE incident_id=? AND customer_id=?",
                  (dumps(m.evidence), m.amount, m.ref_id, m.reason, incident_id, m.customer_id))
            continue
        if role == "at_risk":
            db.ex("UPDATE incident_members SET role='affected', reason=?, evidence_json=?, amount=?, ref_id=?, "
                  "card_id=?, promoted_at=? WHERE incident_id=? AND customer_id=?",
                  (m.reason, dumps(m.evidence), m.amount, m.ref_id, m.card_id, now, incident_id, m.customer_id))
            promoted.append(m.customer_id)
        else:
            db.insert("incident_members", {
                "incident_id": incident_id, "customer_id": m.customer_id, "role": "affected", "reason": m.reason,
                "evidence_json": dumps(m.evidence), "amount": m.amount, "ref_id": m.ref_id, "card_id": m.card_id,
                "first_seen_at": now,
            })
        new_affected.append(m.customer_id)
    for m in impact.at_risk:
        if m.customer_id not in existing:
            db.insert("incident_members", {
                "incident_id": incident_id, "customer_id": m.customer_id, "role": "at_risk", "reason": m.reason,
                "evidence_json": dumps(m.evidence), "amount": m.amount, "ref_id": m.ref_id, "card_id": m.card_id,
                "first_seen_at": now,
            })
    counts = {r["role"]: r["n"] for r in db.q(
        "SELECT role, COUNT(*) AS n FROM incident_members WHERE incident_id=? GROUP BY role", (incident_id,))}
    if promoted:
        log.info("incident %s: at_risk -> affected %s", incident_id, promoted)
    return SyncResult(new_affected, promoted, counts.get("affected", 0), counts.get("at_risk", 0))


def member(db: Database, incident_id: int, customer_id: int) -> dict | None:
    row = db.q1("SELECT * FROM incident_members WHERE incident_id=? AND customer_id=?", (incident_id, customer_id))
    return dict(row) if row else None


def severity(category: str, affected: int, at_risk: int, amount: float) -> str:
    """Severity grows with confirmed impact, potential impact and money involved."""
    score = affected * 1.0 + at_risk * 0.3 + amount / 2000
    if category == "phishing":
        score *= 1.5
    if score >= 25:
        return "critical"
    if score >= 12:
        return "high"
    if score >= 5:
        return "medium"
    return "low"


def bug_priority(sev: str) -> str:
    return {"critical": "P0", "high": "P1", "medium": "P2"}.get(sev, "P3")
