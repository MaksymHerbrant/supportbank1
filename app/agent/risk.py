"""Deterministic customer risk profile (no LLM).

Risk is computed ONLY from financial behaviour and account security. Personal
attributes (age, gender, address, nationality, ...) are deliberately absent
from ``CustomerProfile`` and must never be added.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.db.repo import Database, ago, parse_ts

# Points per factor; tune here. Levels: 0-29 low, 30-59 medium, 60+ high.
RISK_WEIGHTS: dict[str, int] = {
    "refunds_30d_3plus": 30,
    "refund_ratio_over_10pct": 20,
    "rejected_claim_90d": 25,
    "lost_dispute_or_chargeback": 30,
    "tenure_under_30d": 15,
    "device_or_phone_changed_48h": 40,
    "credit_overdue": 20,
    "credit_overdue_over_30d": 40,
    "contacts_7d_3plus": 10,
    "claim_data_mismatch": 25,
}
LEVEL_MEDIUM = 30
LEVEL_HIGH = 60
FORBIDDEN_PERSONAL_FIELDS = frozenset({
    "age", "birth_date", "gender", "sex", "address", "city", "region", "nationality", "ethnicity",
    "religion", "marital_status", "language", "full_name", "phone",
})


class CustomerProfile(BaseModel):
    customer_id: int
    refunds_30d_count: int = 0
    refunds_30d_sum: float = 0.0
    refunds_90d_count: int = 0
    refunds_90d_sum: float = 0.0
    refund_ratio: float = 0.0
    rejected_claims_90d: int = 0
    lost_disputes: int = 0
    chargebacks: int = 0
    tenure_days: int = 0
    device_or_phone_changed_48h: bool = False
    fraud_flags: int = 0
    kyc_complete: bool = True
    has_credit: bool = False
    credit_balance: float = 0.0
    overdue_days: int = 0
    missed_payments_6m: int = 0
    credit_utilization: float = 0.0
    in_restructuring: bool = False
    contacts_7d: int = 0
    claim_data_mismatch: bool = False


class RiskFactor(BaseModel):
    name: str
    points: int
    evidence: str


class RiskAssessment(BaseModel):
    score: int
    level: str
    factors: list[RiskFactor] = Field(default_factory=list)


def build_customer_profile(db: Database, customer_id: int, claim_data_mismatch: bool = False) -> CustomerProfile:
    s = lambda sql, p: db.scalar(sql, p) or 0  # noqa: E731
    d30, d90 = ago(days=30), ago(days=90)
    purchases_90d = float(s("SELECT SUM(amount) FROM transactions WHERE customer_id=? AND created_at>=? "
                            "AND status IN ('settled','disputed')", (customer_id, d90)))
    refunds_90d_sum = float(s("SELECT SUM(amount) FROM refunds WHERE customer_id=? AND status='approved' "
                              "AND created_at>=?", (customer_id, d90)))
    created = db.scalar("SELECT created_at FROM customers WHERE id=?", (customer_id,))
    tenure = (datetime.now(timezone.utc) - parse_ts(created)).days if created else 0
    credit = db.q1("SELECT * FROM credit_accounts WHERE customer_id=?", (customer_id,))
    return CustomerProfile(
        customer_id=customer_id,
        refunds_30d_count=s("SELECT COUNT(*) FROM refunds WHERE customer_id=? AND status='approved' AND created_at>=?",
                            (customer_id, d30)),
        refunds_30d_sum=float(s("SELECT SUM(amount) FROM refunds WHERE customer_id=? AND status='approved' "
                                "AND created_at>=?", (customer_id, d30))),
        refunds_90d_count=s("SELECT COUNT(*) FROM refunds WHERE customer_id=? AND status='approved' AND created_at>=?",
                            (customer_id, d90)),
        refunds_90d_sum=refunds_90d_sum,
        refund_ratio=round(refunds_90d_sum / purchases_90d, 3) if purchases_90d else 0.0,
        rejected_claims_90d=s("SELECT COUNT(*) FROM refunds WHERE customer_id=? AND status='rejected' "
                              "AND created_at>=?", (customer_id, d90)),
        lost_disputes=s("SELECT COUNT(*) FROM disputes WHERE customer_id=? AND outcome='lost' AND kind='dispute'",
                        (customer_id,)),
        chargebacks=s("SELECT COUNT(*) FROM disputes WHERE customer_id=? AND kind='chargeback'", (customer_id,)),
        tenure_days=tenure,
        device_or_phone_changed_48h=bool(s("SELECT COUNT(*) FROM devices WHERE customer_id=? AND changed_at>=?",
                                           (customer_id, ago(hours=48)))),
        fraud_flags=s("SELECT COUNT(*) FROM transactions WHERE customer_id=? AND is_fraud=1 AND created_at>=?",
                      (customer_id, d30)),
        kyc_complete=bool(db.scalar("SELECT kyc_complete FROM customers WHERE id=?", (customer_id,)) or 0),
        has_credit=credit is not None,
        credit_balance=float(credit["balance"]) if credit else 0.0,
        overdue_days=int(credit["overdue_days"]) if credit else 0,
        missed_payments_6m=int(credit["missed_payments_6m"]) if credit else 0,
        credit_utilization=round(credit["balance"] / credit["credit_limit"], 2) if credit and credit["credit_limit"] else 0.0,
        in_restructuring=bool(credit["in_restructuring"]) if credit else False,
        contacts_7d=s("SELECT COUNT(*) FROM tickets WHERE customer_id=? AND created_at>=?", (customer_id, ago(days=7))),
        claim_data_mismatch=claim_data_mismatch,
    )


def risk_score(p: CustomerProfile) -> RiskAssessment:
    """Weighted sum of factors, each with the evidence that triggered it."""
    factors: list[RiskFactor] = []

    def add(name: str, evidence: str) -> None:
        factors.append(RiskFactor(name=name, points=RISK_WEIGHTS[name], evidence=evidence))

    if p.refunds_30d_count >= 3:
        add("refunds_30d_3plus", f"{p.refunds_30d_count} повернень за 30 днів на {p.refunds_30d_sum:.0f} грн")
    if p.refund_ratio > 0.10:
        add("refund_ratio_over_10pct", f"повернення = {p.refund_ratio:.0%} від покупок за 90 днів")
    if p.rejected_claims_90d:
        add("rejected_claim_90d", f"{p.rejected_claims_90d} відхилених претензій за 90 днів")
    if p.lost_disputes or p.chargebacks:
        add("lost_dispute_or_chargeback", f"програні диспути: {p.lost_disputes}, chargeback: {p.chargebacks}")
    if p.tenure_days < 30:
        add("tenure_under_30d", f"клієнт {p.tenure_days} днів")
    if p.device_or_phone_changed_48h:
        add("device_or_phone_changed_48h", "зміна телефону/пристрою за останні 48 год")
    if p.overdue_days > 30:
        add("credit_overdue_over_30d", f"прострочка {p.overdue_days} днів")
    elif p.overdue_days > 0:
        add("credit_overdue", f"прострочка {p.overdue_days} днів")
    if p.contacts_7d >= 3:
        add("contacts_7d_3plus", f"{p.contacts_7d} звернень за 7 днів")
    if p.claim_data_mismatch:
        add("claim_data_mismatch", "заява клієнта не збігається з даними БД")
    score = sum(f.points for f in factors)
    level = "high" if score >= LEVEL_HIGH else "medium" if score >= LEVEL_MEDIUM else "low"
    return RiskAssessment(score=score, level=level, factors=factors)


def assess(db: Database, customer_id: int, claim_data_mismatch: bool = False) -> tuple[CustomerProfile, RiskAssessment]:
    profile = build_customer_profile(db, customer_id, claim_data_mismatch)
    return profile, risk_score(profile)
