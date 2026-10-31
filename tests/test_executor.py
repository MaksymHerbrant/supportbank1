"""Approved plans: audit log and idempotency."""
from __future__ import annotations

import json

from tests.conftest import run_double_charge


async def test_audit_log_written_with_reason_evidence_approver(container) -> None:  # noqa: ANN001
    incident_id = await run_double_charge(container)
    plan_id = container.db.scalar("SELECT id FROM plans WHERE incident_id=?", (incident_id,))
    report = await container.executor.approve_plan(plan_id, "op-anna")
    assert report.failed == 0 and report.notified == 15
    rows = container.db.q("SELECT * FROM actions WHERE action_type='refund_transaction'")
    assert len(rows) == 15
    for r in rows:
        assert r["approved_by"] == "op-anna"
        assert r["actor"] == "operator"
        assert r["reason"].startswith(f"incident #{incident_id}")
        assert json.loads(r["evidence_json"])
        assert r["created_at"]


async def test_second_approval_is_idempotent(container) -> None:  # noqa: ANN001
    incident_id = await run_double_charge(container)
    plan_id = container.db.scalar("SELECT id FROM plans WHERE incident_id=?", (incident_id,))
    first = await container.executor.approve_plan(plan_id, "op")
    count = container.db.scalar("SELECT COUNT(*) FROM actions")
    second = await container.executor.approve_plan(plan_id, "op")
    assert not first.already_processed and second.already_processed
    assert container.db.scalar("SELECT COUNT(*) FROM actions") == count


async def test_plan_and_escalation_cannot_double_refund(container) -> None:  # noqa: ANN001
    # Customer 1 first gets a single-customer escalation with a proposed refund...
    reply = await container.pipeline.handle(1, "Мене двічі списали на АЗС Fuel+")
    esc_id = container.db.scalar("SELECT id FROM escalations WHERE ticket_id=?", (reply.ticket_id,))
    container.db.ex("UPDATE escalations SET status='open' WHERE id=?", (esc_id,))
    await container.executor.decide_escalation(esc_id, True, "op")
    # ...later the same duplicate is part of an incident plan.
    await container.pipeline.handle(2, "Fuel+ зняли гроші два рази")
    await container.pipeline.handle(3, "Подвійне списання fuel plus")
    plan_id = container.db.scalar("SELECT id FROM plans")
    await container.executor.approve_plan(plan_id, "op")
    dup = container.db.scalar("SELECT COUNT(*) FROM actions WHERE action_type='refund_transaction' AND customer_id=1")
    assert dup == 1


async def test_rejected_plan_executes_nothing(container) -> None:  # noqa: ANN001
    incident_id = await run_double_charge(container)
    plan_id = container.db.scalar("SELECT id FROM plans WHERE incident_id=?", (incident_id,))
    assert container.executor.reject_plan(plan_id, "op")
    report = await container.executor.approve_plan(plan_id, "op")
    assert report.already_processed
    assert container.db.scalar("SELECT COUNT(*) FROM actions WHERE action_type='refund_transaction'") == 0


async def test_at_risk_never_receive_messages(container, notifier) -> None:  # noqa: ANN001
    incident_id = await run_double_charge(container)
    plan_id = container.db.scalar("SELECT id FROM plans WHERE incident_id=?", (incident_id,))
    await container.executor.approve_plan(plan_id, "op")
    at_risk = {r[0] for r in container.db.q(
        "SELECT customer_id FROM incident_members WHERE incident_id=? AND role='at_risk'", (incident_id,))}
    assert at_risk
    assert not at_risk & {cid for cid, _ in notifier.customer_messages}
