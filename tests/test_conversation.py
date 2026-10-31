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
