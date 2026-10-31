"""Money decisions: risk profile + LLM proposal + policy.enforce (only ever stricter)."""
from __future__ import annotations

import itertools

import pytest

from app.agent import policy
from app.agent.decision import (
    FORBIDDEN_CUSTOMER_PHRASES, DecisionContext, LLMDecision, ProposedAction, customer_message_ok,
    customer_template, enforce,
)
from app.agent.risk import FORBIDDEN_PERSONAL_FIELDS, CustomerProfile, RiskAssessment, assess
from app.llm import LLMClient, LLMResponse, ToolCall
from tests.conftest import FakeBackend, make_container


def llm_says(decision: str, message: str = "Дякуємо, ми опрацювали ваше звернення.") -> LLMClient:
    """Fake Gemma: no triage output (rules are used), fixed decision for submit_decision."""
    def respond(messages, tools):  # noqa: ANN001, ANN202
        if tools and tools[0].name == "submit_decision":
            return LLMResponse(tool_calls=[ToolCall("submit_decision", {
                "decision": decision, "action": None, "reasoning": "test", "risk_factors_considered": [],
                "operator_note": "нотатка", "customer_message": message})])
        return LLMResponse(text="")
    return LLMClient(FakeBackend(respond))


def ctx(kind: str = "refund_duplicate", level: str = "low", overdue: int = 0, confirmed: bool = True,
        amount: float = 100.0, flags: list[str] | None = None) -> DecisionContext:
    c = DecisionContext(kind=kind, customer_id=1, ticket_id=None, claim_text="x", data_confirmed=confirmed,
                        proposed=None if kind == "credit_relief" else ProposedAction(
                            type="refund_credit_charge" if kind == "credit_fee" else "refund_transaction",
                            ref_id=1, amount=amount),
                        risk_flags=flags or [])
    c.profile = CustomerProfile(customer_id=1, overdue_days=overdue, has_credit=overdue > 0)
    c.risk = RiskAssessment(score={"low": 0, "medium": 40, "high": 80}[level], level=level)
    return c


def decision(d: str) -> LLMDecision:
    return LLMDecision(decision=d, customer_message="ok")


def test_enforce_never_relaxes_needs_operator() -> None:
    for kind, level, overdue, confirmed in itertools.product(
            ["refund_duplicate", "credit_fee", "credit_relief"], ["low", "medium", "high"], [0, 15], [True, False]):
        fd = enforce(decision("needs_operator"), ctx(kind, level, overdue, confirmed))
        assert fd.decision == "needs_operator"


def test_auto_for_medium_risk_is_forced_to_operator() -> None:
    fd = enforce(decision("auto_approve"), ctx(level="medium"))
    assert fd.decision == "needs_operator" and any("ризик medium" in e for e in fd.enforced)


def test_auto_allowed_only_when_everything_is_clean() -> None:
    assert enforce(decision("auto_approve"), ctx()).decision == "auto_approve"
    assert enforce(decision("auto_approve"), ctx(amount=5000)).decision == "needs_operator"
    assert enforce(decision("auto_approve"), ctx(confirmed=False)).decision == "needs_operator"
    assert enforce(decision("auto_approve"), ctx(flags=["angry"])).decision == "needs_operator"
    assert enforce(None, ctx(), llm_available=False).decision == "needs_operator"  # circuit breaker


def test_credit_compensation_with_overdue_needs_operator() -> None:
    fd = enforce(decision("auto_approve"), ctx("credit_fee", overdue=15))
    assert fd.decision == "needs_operator" and any("прострочка" in e for e in fd.enforced)


def test_credit_relief_always_needs_operator() -> None:
    assert enforce(decision("auto_approve"), ctx("credit_relief")).decision == "needs_operator"


def test_high_risk_sets_security_review() -> None:
    fd = enforce(decision("auto_approve"), ctx(level="high"))
    assert fd.security_review and fd.decision == "needs_operator"


@pytest.mark.parametrize("text", [
    "Ми підозрюємо зловживання", "Ваш risk score високий", "через часті повернення ми перевіримо",
    "Ваша історія повернень викликає питання", "фактори ризику",
])
def test_forbidden_customer_phrases_detected(text: str) -> None:
    assert not customer_message_ok(text)


def test_templates_are_neutral() -> None:
    for kind, d in itertools.product(["refund_duplicate", "credit_fee", "credit_relief"],
                                     ["auto_approve", "needs_operator", "info_only"]):
        c = ctx(kind)
        fd = enforce(decision(d), c)
        for executed in (True, False):
            text = customer_template(fd, c, executed and fd.decision == "auto_approve")
            assert customer_message_ok(text), text
            assert policy.is_reply_safe(text, refund_executed=executed and fd.decision == "auto_approve")


def test_profile_has_no_personal_attributes() -> None:
    assert not set(CustomerProfile.model_fields) & FORBIDDEN_PERSONAL_FIELDS


PERSONA_CASES = [
    # customer, message, expected decision, security_review, money moved automatically
    (101, "Таксі зняло з мене гроші двічі за одну поїздку", "auto_approve", False, True),
    (102, "У кіно двічі списали за квиток", "needs_operator", True, False),
    (103, "В Колоску двічі зняли оплату", "needs_operator", True, False),
    (104, "Мені нарахували незрозумілу комісію 250 грн", "needs_operator", False, False),
    (105, "Звідки комісія 120 грн за SMS? Я не підключала", "auto_approve", False, True),
    (106, "Можна відстрочку платежу по кредиту?", "needs_operator", False, False),
]


@pytest.mark.parametrize("cid,text,expected,security,auto_money", PERSONA_CASES)
async def test_each_persona_gets_expected_decision(db, notifier, cid, text, expected, security,  # noqa: ANN001
                                                   auto_money) -> None:
    # The fake LLM is maximally optimistic: it always says auto_approve. Policy must still hold.
    c = make_container(db, notifier, llm=llm_says("auto_approve", "Готово, все вирішено!"))
    reply = await c.pipeline.handle(cid, text)
    d = db.q1("SELECT * FROM decisions WHERE customer_id=? ORDER BY id DESC LIMIT 1", (cid,))
    assert d is not None, reply.triage
    assert d["final_decision"] == expected
    assert bool(d["security_review"]) == security
    moved = db.scalar("SELECT COUNT(*) FROM actions WHERE customer_id=? AND actor='agent' "
                      "AND action_type IN ('refund_transaction','refund_credit_charge')", (cid,))
    assert bool(moved) == auto_money
    assert reply.escalated == (expected == "needs_operator")
    assert customer_message_ok(reply.text)
    assert policy.is_reply_safe(reply.text, refund_executed=auto_money)


async def test_refund_to_credit_card_with_debt_explains_debt(db, notifier) -> None:  # noqa: ANN001
    c = make_container(db, notifier, llm=llm_says("auto_approve", ""))
    reply = await c.pipeline.handle(105, "Звідки комісія 120 грн за SMS? Я не підключала")
    assert "заборгован" in reply.text


async def test_llm_message_revealing_suspicion_is_replaced(db, notifier) -> None:  # noqa: ANN001
    c = make_container(db, notifier, llm=llm_says("needs_operator", "Через часті повернення та підозру ми перевіримо вас"))
    reply = await c.pipeline.handle(102, "У кіно двічі списали за квиток")
    assert customer_message_ok(reply.text)
    for phrase in ("підозр", "часті повернення"):
        assert phrase not in reply.text.lower()


def test_risk_profiles_on_seed(db) -> None:  # noqa: ANN001
    assert assess(db, 101)[1].level == "low"
    assert assess(db, 102)[1].level == "high"
    assert assess(db, 103)[1].level == "medium"
    names = {f.name for f in assess(db, 103)[1].factors}
    assert "device_or_phone_changed_48h" in names
    assert FORBIDDEN_CUSTOMER_PHRASES  # list is configured
