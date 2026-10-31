"""Link new tickets to known incidents and answer with the customer's own status.

Candidate search is hybrid (exact signature, vector similarity, category-only
fallback); whether the customer is affected is decided by SQL in ``impact``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from app.agent.responder import ReplyFacts
from app.agent.triage import Signature, TriageResult
from app.db.repo import Database, local_time, loads, ts, utcnow
from app.incidents import impact
from app.incidents.planner import Planner
from app.vector.store import VectorStore

log = logging.getLogger(__name__)
RESOLVED_GRACE_H = 24
CATEGORY_ONLY_SCORE = 0.75
ESCALATE_FLAGS = {"fraud", "legal", "angry", "repeat_contact", "social_engineering", "vip_pressure", "unverified_claim"}


@dataclass
class LinkCandidate:
    incident_id: int
    score: float
    method: str  # signature | vector | category


@dataclass
class LinkOutcome:
    incident_id: int
    branch: str  # a | b | c | d
    facts: ReplyFacts
    escalate: bool = False
    newly_affected: bool = False


def _no_evidence_hint(category: str) -> str:
    return {
        "double_charge": "дубля транзакції в даних немає",
        "app_payment_error": "невдалих оплат у логах застосунку немає",
        "atm_capture": "захоплення вашої картки банкоматом не зафіксовано",
        "phishing": "шахрайських списань з вашої картки немає",
    }.get(category, "підтвердження в даних немає")


class Linker:
    def __init__(self, db: Database, store: VectorStore, planner: Planner, similarity_threshold: float) -> None:
        self.db = db
        self.store = store
        self.planner = planner
        self.threshold = similarity_threshold

    def _candidate_incidents(self) -> list[dict]:
        cutoff = ts(utcnow() - timedelta(hours=RESOLVED_GRACE_H))
        rows = self.db.q("SELECT * FROM incidents WHERE status='open' OR (status='resolved' AND resolved_at >= ?)",
                         (cutoff,))
        return [dict(r) for r in rows]

    async def find_candidate(self, sig: Signature, text: str) -> LinkCandidate | None:
        incidents = {i["id"]: i for i in self._candidate_incidents()}
        if not incidents:
            return None
        best: LinkCandidate | None = None

        def consider(c: LinkCandidate) -> None:
            nonlocal best
            if best is None or c.score > best.score:
                best = c

        key = sig.key()
        for inc in incidents.values():
            inc_sig = Signature.model_validate_json(inc["signature_json"])
            if key and inc_sig.key() == key:
                consider(LinkCandidate(inc["id"], 1.0, "signature"))
        for item_id, score, meta in await self.store.search("incidents", text, k=3):
            inc = incidents.get(int(item_id))
            if inc is None or score < self.threshold:
                continue
            inc_key = Signature.model_validate_json(inc["signature_json"]).key()
            if key and inc_key and key != inc_key:
                continue  # semantically similar, but a different root cause (e.g. another merchant)
            consider(LinkCandidate(inc["id"], round(score, 3), "vector"))
        if best is None and not key and sig.category != "other":
            same_cat = [i for i in incidents.values() if i["category"] == sig.category and i["status"] == "open"]
            if len(same_cat) == 1:
                consider(LinkCandidate(same_cat[0]["id"], CATEGORY_ONLY_SCORE, "category"))
        return best

    def _executed_for(self, incident_id: int, customer_id: int) -> dict:
        rows = self.db.q("SELECT action_type, amount, created_at FROM actions WHERE incident_id=? AND customer_id=? "
                         "AND status='done' ORDER BY created_at", (incident_id, customer_id))
        out: dict = {"refund": 0.0, "refund_at": None, "actions": []}
        labels = {"block_card": "картку заблоковано", "order_card_reissue": "замовлено перевипуск картки",
                  "open_dispute": "відкрито диспут", "notify_customer": "надіслано інструкції",
                  "refund_transaction": None}
        for r in rows:
            if r["action_type"] == "refund_transaction":
                out["refund"] += r["amount"] or 0
                out["refund_at"] = r["created_at"]
            elif labels.get(r["action_type"]):
                out["actions"].append(labels[r["action_type"]])
        return out

    def _prior_unconfirmed(self, incident_id: int, customer_id: int, ticket_id: int) -> bool:
        return bool(self.db.scalar(
            "SELECT COUNT(*) FROM tickets WHERE incident_id=? AND customer_id=? AND link_branch='c' AND id != ?",
            (incident_id, customer_id, ticket_id)))

    def link(self, ticket_id: int, customer_id: int, cand: LinkCandidate, triage: TriageResult,
             problem: str) -> LinkOutcome:
        """Attach the ticket and decide branch a/b/c/d from data."""
        inc = dict(self.db.q1("SELECT * FROM incidents WHERE id=?", (cand.incident_id,)))
        newly = False
        if inc["status"] == "open":
            sync = impact.sync_members(self.db, inc["id"])
            newly = customer_id in sync.new_affected
            if sync.new_affected:
                self.planner.plan_new_affected(inc["id"])
        mem = impact.member(self.db, inc["id"], customer_id)
        affected = mem is not None and mem["role"] == "affected"
        done = self._executed_for(inc["id"], customer_id)
        facts = ReplyFacts(branch="", language=triage.language, problem=problem)
        escalate = False
        if inc["status"] == "resolved":
            branch = "d"
            facts.branch = "linked_resolved"
            if affected and done["refund"]:
                facts.refund_executed = True
                facts.details = f"кошти {done['refund']:.0f} грн повернуто о {local_time(done['refund_at'])}"
            elif affected:
                facts.details = ", ".join(done["actions"]) or "ваш випадок опрацьовано"
            else:
                facts.details = f"у вашому випадку {_no_evidence_hint(inc['category'])}"
        elif affected and (done["refund"] or (done["actions"] and inc["category"] != "double_charge")):
            branch = "a"
            if done["refund"]:
                facts.branch, facts.amount, facts.refund_executed = "linked_done", done["refund"], True
                facts.refund_time = local_time(done["refund_at"])
            else:
                facts.branch, facts.actions_done = "linked_done_noamount", done["actions"]
        elif affected:
            branch = "b"
            facts.branch = "linked_pending" if inc["category"] == "double_charge" else "linked_pending_noamount"
            self.planner.plan_new_affected(inc["id"])
        else:
            branch = "c"
            facts.details = _no_evidence_hint(inc["category"])
            insists = self._prior_unconfirmed(inc["id"], customer_id, ticket_id)
            escalate = insists or bool(set(triage.risk_flags) & ESCALATE_FLAGS)
            facts.branch = "linked_unconfirmed_escalated" if escalate else "linked_unconfirmed"
        status = "escalated" if escalate else ("waiting_customer" if branch == "c" else "linked")
        self.db.ex("UPDATE tickets SET incident_id=?, link_score=?, link_method=?, link_branch=?, status=? WHERE id=?",
                   (inc["id"], cand.score, cand.method, branch, status, ticket_id))
        log.info("ticket %s linked to incident %s via %s (%.2f) -> branch %s", ticket_id, inc["id"], cand.method,
                 cand.score, branch)
        return LinkOutcome(inc["id"], branch, facts, escalate=escalate, newly_affected=newly)


def linked_counts(db: Database, incident_id: int) -> dict[str, int]:
    rows = db.q("SELECT link_method, COUNT(*) AS n FROM tickets WHERE incident_id=? GROUP BY link_method", (incident_id,))
    return {r["link_method"] or "detector": r["n"] for r in rows}


def signature_of(db: Database, incident_id: int) -> Signature:
    return Signature.model_validate(loads(db.scalar("SELECT signature_json FROM incidents WHERE id=?", (incident_id,)), {}))
