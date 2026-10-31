"""Append-only audit log of every action, with idempotency keys."""
from __future__ import annotations

from typing import Any

from app.db.repo import Database, dumps, now_ts


class AuditLog:
    def __init__(self, db: Database) -> None:
        self.db = db

    def exists(self, idempotency_key: str) -> bool:
        return self.db.q1("SELECT 1 FROM actions WHERE idempotency_key=?", (idempotency_key,)) is not None

    def record(
        self, *, action_type: str, reason: str, actor: str, idempotency_key: str,
        evidence: list[str] | None = None, customer_id: int | None = None, incident_id: int | None = None,
        ticket_id: int | None = None, plan_item_id: int | None = None, params: dict[str, Any] | None = None,
        amount: float | None = None, approved_by: str | None = None, result: dict[str, Any] | None = None,
        status: str = "done",
    ) -> tuple[int, bool]:
        """Insert an action row. Returns (action_id, created); created=False means it already happened."""
        cur = self.db.ex(
            "INSERT OR IGNORE INTO actions (action_type, customer_id, incident_id, ticket_id, plan_item_id, "
            "params_json, amount, reason, evidence_json, actor, approved_by, status, result_json, "
            "idempotency_key, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (action_type, customer_id, incident_id, ticket_id, plan_item_id, dumps(params or {}), amount, reason,
             dumps(evidence or []), actor, approved_by, status, dumps(result) if result else None,
             idempotency_key, now_ts()),
        )
        if cur.rowcount == 0:
            existing = self.db.scalar("SELECT id FROM actions WHERE idempotency_key=?", (idempotency_key,))
            return int(existing), False
        return int(cur.lastrowid), True
