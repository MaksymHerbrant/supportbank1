"""Narrow banking tools. Every mutating tool passes through ``policy`` and the audit log."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.actions.audit import AuditLog
from app.agent import policy
from app.agent.policy import ActionContext, PolicyViolation
from app.db.repo import Database, ago, dumps, now_ts
from app.llm import ToolSpec
from app.notify import Notifier

log = logging.getLogger(__name__)
AUTO_APPROVER_PREFIX = "auto-policy:"


@dataclass
class ToolResult:
    ok: bool
    created: bool = True          # False = idempotent replay, nothing new happened
    action_id: int | None = None
    detail: str = ""


class BankTools:
    """Read tools return plain dicts; action tools return ToolResult and are audited."""

    def __init__(self, db: Database, notifier: Notifier) -> None:
        self.db = db
        self.notifier = notifier
        self.audit = AuditLog(db)

    # ------------------------------------------------------------ read tools

    def get_customer(self, customer_id: int) -> dict[str, Any] | None:
        row = self.db.q1("SELECT id, full_name, segment, language FROM customers WHERE id=?", (customer_id,))
        return dict(row) if row else None

    def get_cards(self, customer_id: int) -> list[dict[str, Any]]:
        rows = self.db.q(
            "SELECT id, last4, product, status, reissue_ordered FROM cards WHERE customer_id=? ORDER BY id",
            (customer_id,),
        )
        return [dict(r) for r in rows]

    def get_transactions(
        self, customer_id: int | None = None, merchant_id: str | None = None, since_hours: float = 24, limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = ("SELECT t.id, t.customer_id, t.card_id, t.merchant_id, m.name AS merchant, t.amount, t.status, "
               "t.is_fraud, t.created_at FROM transactions t JOIN merchants m ON m.id=t.merchant_id "
               "WHERE t.created_at >= ?")
        params: list[Any] = [ago(hours=since_hours)]
        if customer_id is not None:
            sql += " AND t.customer_id=?"
            params.append(customer_id)
        if merchant_id:
            sql += " AND t.merchant_id=?"
            params.append(merchant_id)
        sql += " ORDER BY t.created_at DESC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in self.db.q(sql, params)]

    def get_app_events(
        self, customer_id: int | None = None, app_version: str | None = None, event_type: str | None = None,
        since_hours: float = 24, limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM app_events WHERE created_at >= ?"
        params: list[Any] = [ago(hours=since_hours)]
        for col, val in (("customer_id", customer_id), ("app_version", app_version), ("event_type", event_type)):
            if val is not None:
                sql += f" AND {col}=?"
                params.append(val)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in self.db.q(sql, params)]

    def get_atm_events(
        self, atm_id: str | None = None, customer_id: int | None = None, since_hours: float = 48, limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM atm_events WHERE created_at >= ?"
        params: list[Any] = [ago(hours=since_hours)]
        for col, val in (("atm_id", atm_id), ("customer_id", customer_id)):
            if val is not None:
                sql += f" AND {col}=?"
                params.append(val)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in self.db.q(sql, params)]

    # ------------------------------------------------------------ helpers

    def _card_owner(self, card_id: int) -> int:
        owner = self.db.scalar("SELECT customer_id FROM cards WHERE id=?", (card_id,))
        if owner is None:
            raise PolicyViolation(f"card {card_id} not found")
        return int(owner)

    def _tx(self, tx_id: int) -> dict[str, Any]:
        row = self.db.q1("SELECT * FROM transactions WHERE id=?", (tx_id,))
        if row is None:
            raise PolicyViolation(f"transaction {tx_id} not found")
        return dict(row)

    def _actor(self, ctx: ActionContext) -> str:
        if ctx.approved_by and ctx.approved_by.startswith(AUTO_APPROVER_PREFIX):
            return "agent"  # auto-approved by policy.enforce (low risk, data-confirmed, within limits)
        return "operator" if ctx.approved_by else "agent"

    # ------------------------------------------------------------ action tools

    def block_card(self, ctx: ActionContext, card_id: int, reason: str, evidence: list[str] | None = None,
                   ticket_id: int | None = None) -> ToolResult:
        owner = self._card_owner(card_id)
        policy.check_tool("block_card", ctx, owner)
        if self.db.scalar("SELECT status FROM cards WHERE id=?", (card_id,)) == "blocked":
            return ToolResult(True, False, None, "already blocked")
        key = f"block_card:{card_id}:{ctx.incident_id or ticket_id or 'self'}"
        action_id, created = self.audit.record(
            action_type="block_card", reason=reason, actor=self._actor(ctx), idempotency_key=key,
            evidence=evidence, customer_id=owner, incident_id=ctx.incident_id, ticket_id=ticket_id,
            params={"card_id": card_id}, approved_by=ctx.approved_by,
        )
        if created:
            self.db.ex("UPDATE cards SET status='blocked' WHERE id=?", (card_id,))
        return ToolResult(True, created, action_id, "card blocked")

    def unblock_card(self, ctx: ActionContext, card_id: int, reason: str) -> ToolResult:
        owner = self._card_owner(card_id)
        policy.check_tool("unblock_card", ctx, owner)
        action_id, created = self.audit.record(
            action_type="unblock_card", reason=reason, actor="operator", approved_by=ctx.approved_by,
            idempotency_key=f"unblock_card:{card_id}:{now_ts()}", customer_id=owner, params={"card_id": card_id},
        )
        self.db.ex("UPDATE cards SET status='active' WHERE id=?", (card_id,))
        return ToolResult(True, created, action_id, "card unblocked")

    def order_card_reissue(self, ctx: ActionContext, card_id: int, reason: str, evidence: list[str] | None = None,
                           ticket_id: int | None = None) -> ToolResult:
        owner = self._card_owner(card_id)
        policy.check_tool("order_card_reissue", ctx, owner)
        action_id, created = self.audit.record(
            action_type="order_card_reissue", reason=reason, actor=self._actor(ctx),
            idempotency_key=f"reissue:{card_id}", evidence=evidence, customer_id=owner,
            incident_id=ctx.incident_id, ticket_id=ticket_id, params={"card_id": card_id},
            approved_by=ctx.approved_by,
        )
        if created:
            self.db.ex("UPDATE cards SET reissue_ordered=1 WHERE id=?", (card_id,))
        return ToolResult(True, created, action_id, "reissue ordered")

    def open_dispute(self, ctx: ActionContext, tx_id: int, reason: str, evidence: list[str] | None = None,
                     ticket_id: int | None = None) -> ToolResult:
        """Opens a dispute. Moves no money."""
        tx = self._tx(tx_id)
        policy.check_tool("open_dispute", ctx, tx["customer_id"])
        action_id, created = self.audit.record(
            action_type="open_dispute", reason=reason, actor=self._actor(ctx), idempotency_key=f"dispute:{tx_id}",
            evidence=evidence, customer_id=tx["customer_id"], incident_id=ctx.incident_id, ticket_id=ticket_id,
            params={"tx_id": tx_id}, amount=tx["amount"], approved_by=ctx.approved_by,
        )
        if created and tx["status"] == "settled":
            self.db.ex("UPDATE transactions SET status='disputed' WHERE id=?", (tx_id,))
        return ToolResult(True, created, action_id, "dispute opened")

    def refund_transaction(self, ctx: ActionContext, tx_id: int, reason: str, evidence: list[str] | None = None,
                           plan_item_id: int | None = None) -> ToolResult:
        """Refund. Only with operator approval; at most once per transaction."""
        tx = self._tx(tx_id)
        policy.check_tool("refund_transaction", ctx, tx["customer_id"])
        if tx["status"] == "refunded":
            return ToolResult(True, False, None, "already refunded")
        action_id, created = self.audit.record(
            action_type="refund_transaction", reason=reason, actor=self._actor(ctx), idempotency_key=f"refund:{tx_id}",
            evidence=evidence, customer_id=tx["customer_id"], incident_id=ctx.incident_id,
            plan_item_id=plan_item_id, params={"tx_id": tx_id}, amount=tx["amount"], approved_by=ctx.approved_by,
        )
        if created:
            self.db.ex("UPDATE transactions SET status='refunded' WHERE id=?", (tx_id,))
        return ToolResult(True, created, action_id, f"refunded {tx['amount']:.2f}")

    def refund_credit_charge(self, ctx: ActionContext, charge_id: int, reason: str,
                             evidence: list[str] | None = None) -> ToolResult:
        """Compensate an erroneous credit fee/interest. Requires approval (operator or auto-policy)."""
        row = self.db.q1("SELECT * FROM credit_charges WHERE id=?", (charge_id,))
        if row is None:
            raise PolicyViolation(f"credit charge {charge_id} not found")
        policy.check_tool("refund_credit_charge", ctx, row["customer_id"])
        if row["status"] == "refunded":
            return ToolResult(True, False, None, "already compensated")
        action_id, created = self.audit.record(
            action_type="refund_credit_charge", reason=reason, actor=self._actor(ctx),
            idempotency_key=f"credit_refund:{charge_id}", evidence=evidence, customer_id=row["customer_id"],
            params={"charge_id": charge_id}, amount=row["amount"], approved_by=ctx.approved_by,
        )
        if created:
            self.db.ex("UPDATE credit_charges SET status='refunded' WHERE id=?", (charge_id,))
            self.db.ex("UPDATE credit_accounts SET balance=MAX(0, balance-?) WHERE id=?",
                       (row["amount"], row["account_id"]))
        return ToolResult(True, created, action_id, f"compensated {row['amount']:.2f}")

    def mark_atm_out_of_service(self, ctx: ActionContext, atm_id: str, reason: str,
                                evidence: list[str] | None = None) -> ToolResult:
        if not ctx.approved_by and ctx.level < policy.Level.INVESTIGATE:
            raise PolicyViolation("marking ATM needs investigation level or approval")
        action_id, created = self.audit.record(
            action_type="mark_atm_out_of_service", reason=reason, actor=self._actor(ctx) if ctx.approved_by else "system",
            idempotency_key=f"atm_oos:{atm_id}:{ctx.incident_id}", evidence=evidence, incident_id=ctx.incident_id,
            params={"atm_id": atm_id}, approved_by=ctx.approved_by,
        )
        self.db.ex("UPDATE atms SET status='out_of_service' WHERE id=?", (atm_id,))
        return ToolResult(True, created, action_id, "atm marked out of service")

    async def notify_customer(self, ctx: ActionContext, customer_id: int, text: str, reason: str,
                              idempotency_key: str, evidence: list[str] | None = None) -> ToolResult:
        """Send a message. Allowed only for the requester or confirmed-affected customers."""
        policy.check_tool("notify_customer", ctx, customer_id)
        policy.check_notify(self.db, customer_id, ctx.incident_id, ctx.requester_id)
        text = policy.mask_pan(text) or ""
        action_id, created = self.audit.record(
            action_type="notify_customer", reason=reason, actor=self._actor(ctx), idempotency_key=idempotency_key,
            evidence=evidence, customer_id=customer_id, incident_id=ctx.incident_id,
            params={"text": text}, approved_by=ctx.approved_by,
        )
        if created:
            try:
                await self.notifier.send_customer(customer_id, text)
            except Exception:  # noqa: BLE001 - delivery failure must not break the plan
                log.exception("Delivery to customer %s failed", customer_id)
        return ToolResult(True, created, action_id, "notified")

    async def create_bug_ticket(self, incident_id: int, title: str, body: str, severity: str,
                                priority: str) -> int:
        existing = self.db.scalar("SELECT id FROM bug_tickets WHERE incident_id=?", (incident_id,))
        if existing:
            self.db.ex("UPDATE bug_tickets SET body=?, severity=?, priority=? WHERE id=?",
                       (body, severity, priority, existing))
            return int(existing)
        bug_id = self.db.insert("bug_tickets", {
            "incident_id": incident_id, "title": title, "body": body, "severity": severity,
            "priority": priority, "created_at": now_ts(),
        })
        self.audit.record(action_type="create_bug_ticket", reason=title, actor="agent",
                          idempotency_key=f"bug:{incident_id}", incident_id=incident_id)
        try:
            await self.notifier.dev_ticket(bug_id)
        except Exception:  # noqa: BLE001
            log.exception("Posting bug ticket failed")
        return bug_id

    async def escalate_to_operator(self, ticket_id: int | None, customer_id: int | None, reason: str,
                                   proposed: dict[str, Any] | None = None) -> int:
        esc_id = self.db.insert("escalations", {
            "ticket_id": ticket_id, "customer_id": customer_id, "reason": reason,
            "proposed_json": dumps(proposed) if proposed else None, "created_at": now_ts(),
        })
        if ticket_id:
            self.db.ex("UPDATE tickets SET status='escalated' WHERE id=?", (ticket_id,))
        try:
            await self.notifier.escalation_created(esc_id)
        except Exception:  # noqa: BLE001
            log.exception("Posting escalation failed")
        return esc_id


# Read-only tools exposed to the LLM investigator (native function calling).
INVESTIGATOR_TOOLS = [
    ToolSpec("get_transactions", "List recent transactions, optionally by merchant or customer.", {
        "type": "object", "properties": {
            "merchant_id": {"type": "string"}, "customer_id": {"type": "integer"},
            "since_hours": {"type": "number"}, "limit": {"type": "integer"},
        }}),
    ToolSpec("get_app_events", "List mobile app events (errors, versions).", {
        "type": "object", "properties": {
            "app_version": {"type": "string"}, "event_type": {"type": "string"},
            "customer_id": {"type": "integer"}, "since_hours": {"type": "number"}, "limit": {"type": "integer"},
        }}),
    ToolSpec("get_atm_events", "List ATM events (withdrawals, captured cards, errors).", {
        "type": "object", "properties": {
            "atm_id": {"type": "string"}, "customer_id": {"type": "integer"},
            "since_hours": {"type": "number"}, "limit": {"type": "integer"},
        }}),
]
