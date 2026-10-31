"""Linking new tickets to open incidents and branches a/b/c/d."""
from __future__ import annotations

from app.agent import policy
from app.incidents import impact
from tests.conftest import ConceptEmbedder, make_container, run_double_charge


async def test_paraphrase_links_to_open_incident_via_vector(db, notifier) -> None:  # noqa: ANN001
    c = make_container(db, notifier, embedder=ConceptEmbedder())
    incident_id = await run_double_charge(c)
    reply = await c.pipeline.handle(41, "на заправці зняли гроші двічі")
    assert reply.incident_id == incident_id
    assert reply.link_method == "vector"
    assert reply.link_score >= 0.8


async def test_branch_b_affected_pending_plan(container) -> None:  # noqa: ANN001
    incident_id = await run_double_charge(container)
    reply = await container.pipeline.handle(5, "на заправці зняли гроші двічі")
    assert reply.incident_id == incident_id and reply.branch == "b"
    assert "вже її вирішуємо" in reply.text
    assert policy.is_reply_safe(reply.text, refund_executed=False)
    assert "450" not in reply.text  # no amount promised before approval
    assert container.db.scalar("SELECT COUNT(*) FROM plan_items WHERE customer_id=5 AND status='pending'") >= 1


async def test_branch_a_after_execution_reports_real_status(container) -> None:  # noqa: ANN001
    incident_id = await run_double_charge(container)
    plan_id = container.db.scalar("SELECT id FROM plans WHERE incident_id=? AND status='pending'", (incident_id,))
    await container.executor.approve_plan(plan_id, "operator")
    reply = await container.pipeline.handle(5, "на заправці зняли гроші двічі")
    assert reply.branch == "a"
    assert "450" in reply.text and "повернуто" in reply.text


async def test_branch_c_similar_but_not_confirmed_is_not_affected(db, notifier) -> None:  # noqa: ANN001
    c = make_container(db, notifier, embedder=ConceptEmbedder())
    incident_id = await run_double_charge(c)
    reply = await c.pipeline.handle(41, "на заправці зняли гроші двічі")
    assert reply.branch == "c"
    assert "не бачимо" in reply.text
    mem = impact.member(c.db, incident_id, 41)
    assert mem is None or mem["role"] != "affected"
    assert c.db.scalar("SELECT COUNT(*) FROM plan_items WHERE customer_id=41") == 0


async def test_branch_c_insisting_customer_is_escalated(container) -> None:  # noqa: ANN001
    await run_double_charge(container)
    first = await container.pipeline.handle(41, "на заправці зняли гроші двічі")
    second = await container.pipeline.handle(41, "Я точно бачу два списання, перевірте ще раз!")
    assert first.branch == "c" and not first.escalated
    assert second.branch == "c" and second.escalated


async def test_branch_d_resolved_incident(container) -> None:  # noqa: ANN001
    incident_id = await run_double_charge(container)
    plan_id = container.db.scalar("SELECT id FROM plans WHERE incident_id=?", (incident_id,))
    await container.executor.approve_plan(plan_id, "operator")
    result = await container.incidents.close(incident_id, "operator")
    assert result.ok
    reply = await container.pipeline.handle(6, "двічі зняли на Fuel+")
    assert reply.branch == "d"
    assert "виправлено" in reply.text


async def test_close_sends_final_message_only_to_writers(container, notifier) -> None:  # noqa: ANN001
    incident_id = await run_double_charge(container)
    plan_id = container.db.scalar("SELECT id FROM plans WHERE incident_id=?", (incident_id,))
    await container.executor.approve_plan(plan_id, "operator")
    notifier.customer_messages.clear()
    result = await container.incidents.close(incident_id, "operator")
    recipients = {cid for cid, _ in notifier.customer_messages}
    assert recipients == {1, 2, 3}
    assert result.final_messages == 3


async def test_close_blocked_while_plan_pending(container) -> None:  # noqa: ANN001
    incident_id = await run_double_charge(container)
    result = await container.incidents.close(incident_id, "operator")
    assert not result.ok and "план" in result.message


async def test_exact_signature_works_without_embeddings(container) -> None:  # noqa: ANN001
    incident_id = await run_double_charge(container)
    reply = await container.pipeline.handle(6, "Fuel+ списав двічі")
    assert reply.incident_id == incident_id
    assert reply.link_method == "signature"
