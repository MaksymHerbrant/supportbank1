"""Conversation realism: the bot works out the problem itself; human handoff."""
from __future__ import annotations

from app.bot.live_chat import LiveChats


async def test_greeting_gets_clarifying_question_not_escalation(container) -> None:  # noqa: ANN001
    reply = await container.pipeline.handle(5, "Привіт")
    assert not reply.escalated
    assert "?" in reply.text
    assert "подвійн" not in reply.text.lower()


async def test_generic_refund_request_does_not_assume_the_duplicate(container) -> None:  # noqa: ANN001
    # Customer #5 HAS a Fuel+ duplicate, but asks about something else -> bot must ask, not assume.
    reply = await container.pipeline.handle(5, "Поверніть мені гроші за підписку")
    assert "подвійн" not in reply.text.lower()
    assert "?" in reply.text and not reply.escalated
    assert not container.db.scalar("SELECT COUNT(*) FROM actions WHERE customer_id=5")


async def test_refund_request_naming_the_amount_uses_the_duplicate(container) -> None:  # noqa: ANN001
    reply = await container.pipeline.handle(5, "Поверніть 450 грн, ви зняли двічі")
    assert "450" in reply.text or reply.escalated


async def test_asking_for_a_human_escalates(container) -> None:  # noqa: ANN001
    reply = await container.pipeline.handle(91, "Хочу поговорити з оператором")
    assert reply.escalated and "оператор" in reply.text


async def test_unclear_three_times_escalates(container) -> None:  # noqa: ANN001
    for _ in range(3):
        r = await container.pipeline.handle(91, "ну таке")
        assert not r.escalated
    r = await container.pipeline.handle(91, "ну таке")
    assert r.escalated


async def test_history_is_collected(container) -> None:  # noqa: ANN001
    await container.pipeline.handle(91, "Привіт")
    assert "Клієнт: Привіт" in container.pipeline.history(91)


def test_live_chat_lifecycle(db) -> None:  # noqa: ANN001
    live = LiveChats(db)
    chat, created = live.open(5, "Анна")
    assert created and live.active(5)["operator_name"] == "Анна"
    assert not live.open(5, "Інший")[1]  # already open
    live.remember(777, 5, "to_operator", "картка 4149 4993 1234 5678")
    assert live.customer_of(777) == 5
    assert "4149" not in db.scalar("SELECT text FROM relay_messages WHERE group_message_id=777")
    assert live.close(5) and live.active(5) is None


async def test_large_amount_goes_to_human(container) -> None:  # noqa: ANN001
    reply = await container.pipeline.handle(91, "Поверніть 25 000 грн, які я переказав")
    assert reply.escalated


async def test_angry_customer_is_not_handed_off_for_anger_alone(container) -> None:  # noqa: ANN001
    reply = await container.pipeline.handle(101, "ЖАХ!!! Таксі зняло з мене гроші двічі!!!")
    assert reply.triage.category == "double_charge"
    assert "angry" in reply.triage.risk_flags
    assert reply.triage.level != 4


async def test_dispute_resolved_by_agent(container) -> None:  # noqa: ANN001
    amount = int(container.db.scalar("SELECT amount FROM transactions WHERE customer_id=92 ORDER BY id LIMIT 1"))
    reply = await container.pipeline.handle(92, f"Хочу оскаржити операцію на {amount} грн, я її не впізнаю")
    assert not reply.escalated
    assert any(a.startswith("open_dispute") for a in reply.actions)


async def test_repeat_question_after_refund_says_already_refunded(db, notifier) -> None:  # noqa: ANN001
    from tests.test_decision import llm_says
    from tests.conftest import make_container
    c = make_container(db, notifier, llm=llm_says("auto_approve"))
    first = await c.pipeline.handle(101, "Таксі зняло з мене гроші двічі")
    second = await c.pipeline.handle(101, "Таксі зняло з мене гроші двічі")
    assert first.actions and not second.actions
    assert "уже повернуто" in second.text
    assert db.scalar("SELECT COUNT(*) FROM actions WHERE action_type='refund_transaction' AND customer_id=101") == 1


def test_agent_personas_seeded(db) -> None:  # noqa: ANN001
    from app.incidents import impact
    from app.db.repo import ago
    pairs = impact.duplicate_pairs(db, None, ago(days=1), customer_id=107)
    assert len(pairs) == 1 and pairs[0]["amount"] == 95.0
    assert db.scalar("SELECT COUNT(*) FROM transactions WHERE customer_id=107 AND merchant_id='khvylia'") == 1
    assert db.scalar("SELECT COUNT(*) FROM transactions WHERE customer_id=108 AND is_fraud=1") == 1


async def test_agent_tools_refund_only_real_duplicate(db, notifier) -> None:  # noqa: ANN001
    """Scripted Gemma: asks for the subscription refund first (rejected), then the coffee duplicate (refunded)."""
    from app.llm import LLMClient, LLMResponse, ToolCall
    from tests.conftest import FakeBackend, make_container
    sub_tx = db.scalar("SELECT id FROM transactions WHERE customer_id=107 AND merchant_id='khvylia'")
    dup_tx = db.scalar("SELECT MAX(id) FROM transactions WHERE customer_id=107 AND merchant_id='zerno'")
    script = iter([
        [ToolCall("request_refund", {"tx_id": sub_tx})],
        [ToolCall("request_refund", {"tx_id": dup_tx})],
        [ToolCall("reply_to_customer", {"text": "Повернув дубль 95 грн за каву, підписка — звичайний платіж."})],
    ])

    def respond(messages, tools):  # noqa: ANN001, ANN202
        names = {t.name for t in tools or []}
        if "submit_decision" in names:
            return LLMResponse(tool_calls=[ToolCall("submit_decision", {
                "decision": "auto_approve", "reasoning": "дубль", "customer_message": "ok"})])
        if "reply_to_customer" in names:
            return LLMResponse(tool_calls=next(script))
        return LLMResponse(text="")
    c = make_container(db, notifier, llm=LLMClient(FakeBackend(respond)))
    reply = await c.pipeline.handle(107, "мені щось двічі списали")
    assert "95" in reply.text
    refunded = {r[0] for r in db.q("SELECT json_extract(params_json,'$.tx_id') FROM actions "
                                   "WHERE action_type='refund_transaction'")}
    assert refunded == {sub_tx, dup_tx}
    reasons = {r[0]: r[1] for r in db.q("SELECT json_extract(params_json,'$.tx_id'), reason FROM actions "
                                        "WHERE action_type='refund_transaction'")}
    assert reasons[sub_tx].startswith("goodwill")      # normal charge: goodwill path (<= 1000, low risk)
    assert not reasons[dup_tx].startswith("goodwill")  # real duplicate: bank-error path


def _agent_llm(tool_steps, decision="auto_approve"):  # noqa: ANN001, ANN202
    from app.llm import LLMClient, LLMResponse, ToolCall
    from tests.conftest import FakeBackend
    script = iter(tool_steps)

    def respond(messages, tools):  # noqa: ANN001, ANN202
        names = {t.name for t in tools or []}
        if "reply_to_customer" in names:
            return LLMResponse(tool_calls=next(script))
        if "submit_decision" in names:
            return LLMResponse(tool_calls=[ToolCall("submit_decision", {"decision": decision,
                                                                        "customer_message": "ok"})])
        return LLMResponse(text="")
    return LLMClient(FakeBackend(respond))


async def test_goodwill_refund_up_to_1000_for_low_risk(db, notifier) -> None:  # noqa: ANN001
    from app.llm import ToolCall
    from tests.conftest import make_container
    sub_tx = db.scalar("SELECT id FROM transactions WHERE customer_id=107 AND merchant_id='khvylia'")
    llm = _agent_llm([[ToolCall("request_refund", {"tx_id": sub_tx})],
                      [ToolCall("reply_to_customer", {"text": "Повернув 199 грн за підписку."})]])
    c = make_container(db, notifier, llm=llm)
    reply = await c.pipeline.handle(107, "Поверніть 199 за підписку, вона списалась хоча я скасувала")
    assert not reply.escalated
    assert db.scalar("SELECT status FROM transactions WHERE id=?", (sub_tx,)) == "refunded"
    assert db.scalar("SELECT actor FROM actions WHERE action_type='refund_transaction'") == "agent"


async def test_second_goodwill_within_30_days_goes_to_operator(db, notifier) -> None:  # noqa: ANN001
    from app.llm import ToolCall
    from tests.conftest import make_container
    txs = [r[0] for r in db.q("SELECT id FROM transactions WHERE customer_id=107 AND merchant_id!='zerno' "
                              "AND amount < 1000 ORDER BY id DESC LIMIT 2")]
    llm = _agent_llm([[ToolCall("request_refund", {"tx_id": txs[0]})], [ToolCall("reply_to_customer", {"text": "ok"})],
                      [ToolCall("request_refund", {"tx_id": txs[1]})], [ToolCall("reply_to_customer", {"text": "ok"})]])
    c = make_container(db, notifier, llm=llm)
    first = await c.pipeline.handle(107, "послугу не надали, поверніть")
    second = await c.pipeline.handle(107, "і ще за цю послугу поверніть, її теж не надали")
    assert not first.escalated and second.escalated


async def test_goodwill_over_1000_needs_operator(db, notifier) -> None:  # noqa: ANN001
    from app.llm import ToolCall
    from tests.conftest import make_container
    tx = db.insert("transactions", {"customer_id": 107, "card_id": db.scalar("SELECT id FROM cards WHERE customer_id=107"),
                                    "merchant_id": "shvydko", "amount": 1500.0, "created_at": "2026-10-30 10:00:00"})
    llm = _agent_llm([[ToolCall("request_refund", {"tx_id": tx})], [ToolCall("reply_to_customer", {"text": "ok"})]])
    c = make_container(db, notifier, llm=llm)
    reply = await c.pipeline.handle(107, "таксі не приїхало, поверніть 1500")
    assert reply.escalated
    assert db.scalar("SELECT status FROM transactions WHERE id=?", (tx,)) != "refunded"


async def test_contested_normal_charge_goes_to_operator_and_refunds_on_approval(db, notifier) -> None:  # noqa: ANN001
    """Model says needs_operator -> operator card -> approval refunds and notifies."""
    from app.llm import LLMClient, LLMResponse, ToolCall
    from tests.conftest import FakeBackend, make_container
    sub_tx = db.scalar("SELECT id FROM transactions WHERE customer_id=108 AND merchant_id='khvylia'")
    script = iter([[ToolCall("request_refund", {"tx_id": sub_tx, "reason": "списали після скасування"})],
                   [ToolCall("reply_to_customer", {"text": "Передав запит фахівцю, він підтвердить повернення."})]])

    def respond(messages, tools):  # noqa: ANN001, ANN202
        names = {t.name for t in tools or []}
        if "reply_to_customer" in names:
            return LLMResponse(tool_calls=next(script))
        if "submit_decision" in names:
            return LLMResponse(tool_calls=[ToolCall("submit_decision", {"decision": "needs_operator",
                                                                        "customer_message": "ok"})])
        return LLMResponse(text="")
    c = make_container(db, notifier, llm=LLMClient(FakeBackend(respond)))
    reply = await c.pipeline.handle(108, "Поверніть 199 за підписку, вона списалась хоча я скасував")
    assert reply.escalated
    assert db.scalar("SELECT COUNT(*) FROM actions WHERE action_type='refund_transaction'") == 0  # not auto
    esc = db.q1("SELECT * FROM escalations WHERE customer_id=108 AND proposed_json IS NOT NULL")
    assert db.scalar("SELECT final_decision FROM decisions WHERE customer_id=108") == "needs_operator"
    result = await c.executor.decide_escalation(esc["id"], True, "operator")
    assert "Повернено" in result
    assert db.scalar("SELECT status FROM transactions WHERE id=?", (sub_tx,)) == "refunded"
    assert any(cid == 108 and "повернуто" in t for cid, t in notifier.customer_messages)


async def test_top_and_action_feed(container) -> None:  # noqa: ANN001
    from app import stats
    await container.pipeline.handle(91, "Який ліміт на зняття готівки?")
    await container.pipeline.handle(89, "Мені дзвонили з банку, я продиктував код з SMS")
    text = stats.render_top(container.db, "2000-01-01 00:00:00")
    assert "Топ-3" in text and "block_card" in text
