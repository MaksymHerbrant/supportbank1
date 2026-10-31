"""Full pipeline with an LLM present (scripted Gemma-like backend)."""
from __future__ import annotations

import json

from app.agent import policy
from app.llm import LLMClient, LLMResponse, ToolCall
from tests.conftest import ConceptEmbedder, FakeBackend, make_container


def gemma_like(messages, tools):  # noqa: ANN001, ANN201
    system = messages[0].content
    names = {t.name for t in tools or []}
    if "submit_triage" in names:
        user = messages[-1].content.lower()
        cat = "double_charge" if ("двічі" in user or "два рази" in user) else "faq"
        return LLMResponse(tool_calls=[ToolCall("submit_triage", {
            "intent": "complaint", "category": cat, "level": 3 if cat != "faq" else 1, "confidence": 0.92,
            "risk_flags": [], "merchant": "АЗС Fuel+" if "fuel" in user else None, "language": "uk"})])
    if "submit_hypothesis" in names:
        if not any(m.role == "tool" for m in messages):  # first step: use a read tool
            return LLMResponse(tool_calls=[ToolCall("get_transactions", {"merchant_id": "fuelplus", "limit": 5})])
        tool_out = next(m.content for m in messages if m.role == "tool")
        first_id = json.loads(tool_out.split("-> ", 1)[1])[0]["id"]
        return LLMResponse(tool_calls=[ToolCall("submit_hypothesis", {
            "hypothesis": "Термінал Fuel+ повторює авторизацію після таймауту",
            "evidence": [f"tx #{first_id} повтор", "tx #999999 вигаданий"], "suspected_component": "acquiring"})])
    if "incident summary" in system:
        return LLMResponse(text="Масові подвійні списання на АЗС Fuel+ сьогодні.")
    if "customer-support replies" in system:
        return LLMResponse(text="Ми вже повернули вам гроші!")  # unsafe on purpose -> must be rejected
    return LLMResponse(text="ok")


async def test_pipeline_with_llm(db, notifier) -> None:  # noqa: ANN001
    backend = FakeBackend(gemma_like)
    c = make_container(db, notifier, llm=LLMClient(backend), embedder=ConceptEmbedder())
    for cid, text in [(1, "двічі списали на fuel+"), (2, "fuel+ два рази зняли"), (3, "двічі fuel+ оплата")]:
        reply = await c.pipeline.handle(cid, text)
        assert reply.triage.source == "llm"
        assert policy.is_reply_safe(reply.text, refund_executed=False)
    await c.incidents.drain()
    inc = db.q1("SELECT * FROM incidents")
    assert inc["summary"].startswith("Масові подвійні")
    assert inc["hypothesis"].startswith("Термінал Fuel+")
    evidence = json.loads(inc["evidence_json"])
    assert not any("999999" in e for e in evidence)  # hallucinated id filtered out
    late = await c.pipeline.handle(5, "на заправці двічі зняли гроші")
    assert late.branch == "b"
    assert policy.is_reply_safe(late.text, refund_executed=False)
