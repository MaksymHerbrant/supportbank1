"""Action plans for confirmed-affected customers only (never at_risk)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.db.repo import Database, dumps, loads, now_ts

log = logging.getLogger(__name__)


@dataclass
class PlanPreview:
    plan_id: int
    items: int
    customers: int
    refund_total: float
    lines: list[str]


def _items_for(category: str, member: dict) -> list[tuple[str, str, dict, float | None]]:
    """(action, ref, params, amount) for one affected member."""
    cid = member["customer_id"]
    refs = [r for r in (member["ref_id"] or "").split(",") if r]
    items: list[tuple[str, str, dict, float | None]] = []
    if category == "double_charge":
        for tx_id in refs:
            items.append(("refund_transaction", tx_id, {"tx_id": int(tx_id)}, None))
    elif category == "phishing":
        if member["card_id"]:
            items.append(("block_card", str(member["card_id"]), {"card_id": member["card_id"]}, None))
        for tx_id in refs:
            items.append(("open_dispute", tx_id, {"tx_id": int(tx_id)}, None))
    elif category == "atm_capture" and member["card_id"]:
        items.append(("block_card", str(member["card_id"]), {"card_id": member["card_id"]}, None))
        items.append(("order_card_reissue", str(member["card_id"]), {"card_id": member["card_id"]}, None))
    items.append(("notify_customer", "", {"customer_id": cid}, None))
    return items


class Planner:
    def __init__(self, db: Database) -> None:
        self.db = db

    def _pending_plan(self, incident_id: int) -> int:
        row = self.db.q1("SELECT id FROM plans WHERE incident_id=? AND status='pending' ORDER BY id DESC LIMIT 1",
                         (incident_id,))
        if row:
            return int(row["id"])
        return self.db.insert("plans", {"incident_id": incident_id, "created_at": now_ts()})

    def plan_new_affected(self, incident_id: int) -> int | None:
        """Add every affected member not yet planned into the pending plan. Returns plan id if items were added."""
        inc = self.db.q1("SELECT category, status FROM incidents WHERE id=?", (incident_id,))
        if inc is None or inc["status"] != "open":
            return None
        members = self.db.q(
            "SELECT * FROM incident_members m WHERE incident_id=? AND role='affected' AND NOT EXISTS "
            "(SELECT 1 FROM plan_items p WHERE p.incident_id=m.incident_id AND p.customer_id=m.customer_id)",
            (incident_id,),
        )
        if not members:
            return None
        plan_id = self._pending_plan(incident_id)
        added = 0
        for m in members:
            m = dict(m)
            for action, ref, params, _ in _items_for(inc["category"], m):
                amount = None
                if action == "refund_transaction":
                    amount = self.db.scalar("SELECT amount FROM transactions WHERE id=?", (params["tx_id"],))
                cur = self.db.ex(
                    "INSERT OR IGNORE INTO plan_items (plan_id, incident_id, customer_id, action, ref, params_json, "
                    "amount, reason, evidence_json) VALUES (?,?,?,?,?,?,?,?,?)",
                    (plan_id, incident_id, m["customer_id"], action, ref, dumps(params), amount, m["reason"],
                     m["evidence_json"]),
                )
                added += cur.rowcount
        if added == 0:
            return None
        log.info("incident %s: plan %s +%s items", incident_id, plan_id, added)
        return plan_id

    def preview(self, plan_id: int, limit: int = 6) -> PlanPreview:
        items = self.db.q(
            "SELECT p.*, c.full_name FROM plan_items p LEFT JOIN customers c ON c.id=p.customer_id "
            "WHERE plan_id=? ORDER BY customer_id, id", (plan_id,))
        by_customer: dict[int, list] = {}
        for it in items:
            by_customer.setdefault(it["customer_id"], []).append(it)
        labels = {"refund_transaction": "повернення", "block_card": "блок картки", "order_card_reissue": "перевипуск",
                  "open_dispute": "диспут", "notify_customer": "повідомлення"}
        lines = []
        for cid, its in list(by_customer.items())[:limit]:
            acts = ", ".join(
                f"{labels.get(i['action'], i['action'])}" + (f" {i['amount']:.0f} грн" if i["amount"] else "")
                for i in its)
            ev = loads(its[0]["evidence_json"], [])
            lines.append(f"• #{cid} {its[0]['full_name']}: {acts}\n   ↳ {ev[0] if ev else its[0]['reason']}")
        total = sum(i["amount"] or 0 for i in items if i["action"] == "refund_transaction")
        return PlanPreview(plan_id, len(items), len(by_customer), round(total, 2), lines)
