"""affected / at_risk sets on seed data are decided by SQL only."""
from __future__ import annotations

from datetime import timedelta

from app.agent.triage import Signature
from app.db.repo import ts, utcnow
from app.incidents import impact


def ids(members) -> set[int]:  # noqa: ANN001
    return {m.customer_id for m in members}


def test_double_charge_sets(db) -> None:  # noqa: ANN001
    res = impact.compute(db, Signature(category="double_charge", merchant_id="fuelplus"))
    assert ids(res.affected) == set(range(1, 16))
    assert ids(res.at_risk) == set(range(16, 41))  # #3 has a recurring payment but is affected, #95 is too far
    assert 41 not in ids(res.affected) | ids(res.at_risk)
    m5 = next(m for m in res.affected if m.customer_id == 5)
    assert m5.amount == 450.0 and "інтервал" in m5.evidence[0]


def test_phishing_sets(db) -> None:  # noqa: ANN001
    res = impact.compute(db, Signature(category="phishing", scam_domain="demobank-secure.top"))
    assert ids(res.affected) == {42, 43, 44, 45}
    assert ids(res.at_risk) == {46, 47}


def test_atm_sets(db) -> None:  # noqa: ANN001
    res = impact.compute(db, Signature(category="atm_capture", atm_id="ATM-0417"))
    assert ids(res.affected) == set(range(48, 53))
    assert ids(res.at_risk) == set(range(53, 59))


def test_app_bug_sets(db) -> None:  # noqa: ANN001
    res = impact.compute(db, Signature(category="app_payment_error", app_version="5.4.0", error_code="E-PAY-502"))
    assert ids(res.affected) == set(range(59, 67))
    assert ids(res.at_risk) == set(range(67, 87))


def test_unknown_entity_gives_empty_impact(db) -> None:  # noqa: ANN001
    res = impact.compute(db, Signature(category="double_charge"))
    assert res.affected == [] and res.at_risk == []


def test_at_risk_promoted_to_affected(db) -> None:  # noqa: ANN001
    sig = Signature(category="double_charge", merchant_id="fuelplus")
    inc = db.insert("incidents", {"category": "double_charge", "signature_json": sig.model_dump_json(),
                                  "summary": "t", "created_at": ts(utcnow())})
    first = impact.sync_members(db, inc)
    assert first.affected_count == 15 and first.at_risk_count == 25
    assert impact.member(db, inc, 16)["role"] == "at_risk"
    card = db.scalar("SELECT card_id FROM recurring_payments WHERE customer_id=16")
    now = utcnow()
    for sec in (0, 30):
        db.insert("transactions", {"customer_id": 16, "card_id": card, "merchant_id": "fuelplus", "amount": 750.0,
                                   "created_at": ts(now - timedelta(seconds=60 - sec))})
    second = impact.sync_members(db, inc)
    assert second.promoted == [16]
    assert impact.member(db, inc, 16)["role"] == "affected"
    assert impact.member(db, inc, 16)["promoted_at"] is not None
    assert second.affected_count == 16 and second.at_risk_count == 24


def test_severity_uses_at_risk() -> None:
    assert impact.severity("double_charge", 3, 0, 0) == "low"
    assert impact.severity("double_charge", 3, 30, 0) == "high"
