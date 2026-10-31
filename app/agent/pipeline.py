"""End-to-end handling of one customer message.

1. mask + triage (LLM proposes, policy decides level)
2. linker: known incident? -> instant status reply (branches a/b/c/d)
3. otherwise handle by level (FAQ / safe action / investigation / human)
4. detector: maybe this complaint completes a new incident
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from app.agent import policy
from app.agent.faq_rag import FaqRAG
from app.agent.policy import ActionContext, Level, PolicyViolation
from app.agent.responder import ReplyFacts, Responder
from app.agent.tools import BankTools
from app.agent.triage import INCIDENT_CATEGORIES, MediaFields, Signature, Triage, TriageResult
from app.actions.executor import incident_problem
from app.db.repo import Database, ago, dumps, local_time, now_ts
from app.incidents import impact
from app.incidents.service import IncidentService

log = logging.getLogger(__name__)


@dataclass
class Reply:
    text: str
    ticket_id: int
    triage: TriageResult
    incident_id: int | None = None
    branch: str | None = None
    link_method: str | None = None
    link_score: float | None = None
    new_incident_id: int | None = None
    actions: list[str] = field(default_factory=list)
    escalated: bool = False


class SupportPipeline:
    def __init__(self, db: Database, triage: Triage, tools: BankTools, faq: FaqRAG, responder: Responder,
                 incidents: IncidentService) -> None:
        self.db = db
        self.triage = triage
        self.tools = tools
        self.faq = faq
        self.responder = responder
        self.incidents = incidents
        # One message at a time: incident detection must not race (SQLite + shared state).
        self._lock = asyncio.Lock()

    async def handle(self, customer_id: int, text: str, media: MediaFields | None = None) -> Reply:
        async with self._lock:
            try:
                return await self._handle(customer_id, text, media)
            except Exception:  # noqa: BLE001 - the bot must always answer
                log.exception("pipeline failed for customer %s", customer_id)
                ticket_id = self.db.insert("tickets", {"customer_id": customer_id, "text": text or "(media)",
                                                       "status": "escalated", "created_at": now_ts()})
                await self.tools.escalate_to_operator(ticket_id, customer_id, "internal error while processing")
                tr = TriageResult(intent="error", category="other", level=Level.HUMAN, confidence=0.0,
                                  risk_flags=[], signature=Signature())
                return Reply(await self.responder.compose(ReplyFacts("escalated")), ticket_id, tr, escalated=True)

    async def _handle(self, customer_id: int, text: str, media: MediaFields | None) -> Reply:
        text = policy.mask_pan(text or "") or ""
        if not text and media:
            text = f"(медіа) {media.as_text()}"
        tr = await self.triage.classify(customer_id, text, media)
        ticket_id = self.db.insert("tickets", {
            "customer_id": customer_id, "text": text, "media_json": media.model_dump_json() if media else None,
            "intent": tr.intent, "category": tr.category, "level": int(tr.level), "confidence": tr.confidence,
            "risk_flags_json": dumps(tr.risk_flags), "signature_json": tr.signature.model_dump_json(),
            "created_at": now_ts(),
        })

        # 1) Known incident? Link first, answer with this customer's own status.
        cand = await self.incidents.linker.find_candidate(tr.signature, text)
        if cand is not None:
            problem = incident_problem(self.db, cand.incident_id)
            outcome = self.incidents.linker.link(ticket_id, customer_id, cand, tr, problem)
            reply = await self.responder.compose(outcome.facts)
            esc = False
            if outcome.escalate:
                await self.tools.escalate_to_operator(
                    ticket_id, customer_id, f"incident #{outcome.incident_id}: claim not confirmed by data "
                    f"(flags: {', '.join(tr.risk_flags) or 'repeat'})")
                esc = True
            self._save_reply(ticket_id, reply)
            await self.incidents._changed(outcome.incident_id)
            return Reply(reply, ticket_id, tr, incident_id=outcome.incident_id, branch=outcome.branch,
                         link_method=cand.method, link_score=cand.score, escalated=esc)

        # 2) Normal flow by level.
        reply = await self._by_level(customer_id, ticket_id, text, tr, media)

        # 3) Could this complaint complete a new incident?
        if tr.category in INCIDENT_CATEGORIES:
            await self.incidents.detector.index_complaint(ticket_id, text, tr.category)
            cluster = await self.incidents.detector.find_cluster(ticket_id)
            if cluster:
                reply.new_incident_id = await self.incidents.create_from_cluster(cluster)
                reply.incident_id = reply.new_incident_id
        self._save_reply(ticket_id, reply.text)
        return reply

    def _save_reply(self, ticket_id: int, text: str) -> None:
        self.db.ex("UPDATE tickets SET reply=? WHERE id=?", (text, ticket_id))

    def _set_status(self, ticket_id: int, status: str) -> None:
        self.db.ex("UPDATE tickets SET status=? WHERE id=? AND status NOT IN ('escalated')", (status, ticket_id))

    async def _escalate(self, ticket_id: int, customer_id: int, tr: TriageResult, reason: str,
                        proposed: dict | None = None) -> None:
        await self.tools.escalate_to_operator(ticket_id, customer_id, reason, proposed)

    async def _by_level(self, cid: int, ticket_id: int, text: str, tr: TriageResult,
                        media: MediaFields | None) -> Reply:
        ctx = ActionContext(level=tr.level, requester_id=cid)
        facts = ReplyFacts(branch="escalated", language=tr.language)
        out = Reply("", ticket_id, tr)
        cat = tr.category
        try:
            if cat == "social_engineering" or "social_engineering" in tr.risk_flags:
                blocked = self._block_all_cards(ctx, cid, ticket_id, "social engineering: customer disclosed codes")
                facts = ReplyFacts("social_engineering", tr.language, details=", ".join(blocked) or "")
                out.actions += [f"block_card {c}" for c in blocked]
                await self._escalate(ticket_id, cid, tr, "URGENT social engineering: customer shared codes; cards blocked")
                out.escalated = True
            elif cat == "phishing":
                facts, out = await self._phishing(ctx, cid, ticket_id, tr, facts, out)
            elif tr.level == Level.HUMAN:
                await self._escalate(ticket_id, cid, tr, f"level 4: flags={tr.risk_flags} confidence={tr.confidence}")
                out.escalated = True
            elif cat == "faq":
                ans = await self.faq.answer(text, tr.language)
                if ans.escalate or not ans.text:
                    await self._escalate(ticket_id, cid, tr, "FAQ: no relevant fragment found")
                    out.escalated = True
                else:
                    facts = ReplyFacts("faq", tr.language, details=ans.text)
                    self._set_status(ticket_id, "auto_resolved")
            elif cat in ("atm_capture", "card_lost"):
                card = self._card_for(cid, tr)
                if card is None:
                    await self._escalate(ticket_id, cid, tr, "no active card found to block")
                    out.escalated = True
                else:
                    reason = f"{cat}: requested by card owner"
                    self.tools.block_card(ctx, card["id"], reason, ticket_id=ticket_id)
                    self.tools.order_card_reissue(ctx, card["id"], reason, ticket_id=ticket_id)
                    out.actions += [f"block_card *{card['last4']}", f"order_card_reissue *{card['last4']}"]
                    facts = ReplyFacts("card_blocked_reissue", tr.language, details=f"*{card['last4']}")
                    self._set_status(ticket_id, "auto_resolved")
            elif cat == "double_charge":
                facts, out = await self._double_charge(ctx, cid, ticket_id, tr, media, out)
            elif cat == "app_payment_error":
                fails = self.tools.get_app_events(customer_id=cid, event_type="payment_failed", since_hours=24)
                if fails:
                    f = fails[0]
                    facts = ReplyFacts("app_error", tr.language, details=f"{f['error_code']} (версія {f['app_version']})")
                    self._set_status(ticket_id, "auto_resolved")
                else:
                    facts = ReplyFacts("app_no_error", tr.language)
                    self._set_status(ticket_id, "waiting_customer")
            elif cat == "refund_demand":
                dups = impact.duplicate_pairs(self.db, None, ago(days=30), customer_id=cid)
                if dups:
                    facts, out = await self._double_charge(ctx, cid, ticket_id, tr, media, out)
                else:
                    facts = ReplyFacts("refund_unfounded", tr.language,
                                       details="дублів чи помилкових списань за 30 днів немає")
                    await self._escalate(ticket_id, cid, tr, f"refund demand without grounds in data; flags={tr.risk_flags}")
                    out.escalated = True
            else:
                await self._escalate(ticket_id, cid, tr, f"category={cat}, level={int(tr.level)}")
                out.escalated = True
        except PolicyViolation as exc:
            log.warning("policy blocked action for ticket %s: %s", ticket_id, exc)
            await self._escalate(ticket_id, cid, tr, f"policy blocked automatic action: {exc}")
            out.escalated = True
            facts = ReplyFacts("escalated", tr.language)
        out.text = await self.responder.compose(facts)
        return out

    def _card_for(self, cid: int, tr: TriageResult) -> dict | None:
        if tr.signature.atm_id:
            row = self.db.q1("SELECT c.* FROM atm_events e JOIN cards c ON c.id=e.card_id WHERE e.customer_id=? "
                             "AND e.atm_id=? AND e.event_type='card_captured' ORDER BY e.created_at DESC LIMIT 1",
                             (cid, tr.signature.atm_id))
            if row:
                return dict(row)
        row = self.db.q1("SELECT * FROM cards WHERE customer_id=? ORDER BY status='active' DESC, id LIMIT 1", (cid,))
        return dict(row) if row else None

    def _block_all_cards(self, ctx: ActionContext, cid: int, ticket_id: int, reason: str) -> list[str]:
        blocked = []
        for card in self.tools.get_cards(cid):
            if card["status"] == "active":
                self.tools.block_card(ctx, card["id"], reason, ticket_id=ticket_id)
                blocked.append(f"*{card['last4']}")
        return blocked

    async def _phishing(self, ctx: ActionContext, cid: int, ticket_id: int, tr: TriageResult, facts: ReplyFacts,
                        out: Reply) -> tuple[ReplyFacts, Reply]:
        fraud = [t for t in self.tools.get_transactions(customer_id=cid, since_hours=72) if t["is_fraud"]]
        if fraud or "fraud" in tr.risk_flags:
            blocked = self._block_all_cards(ctx, cid, ticket_id, "phishing: confirmed fraudulent transaction"
                                            if fraud else "phishing: customer reports compromise")
            out.actions += [f"block_card {c}" for c in blocked]
            proposed = {"action": "open_dispute", "tx_ids": [t["id"] for t in fraud]} if fraud else None
            await self._escalate(ticket_id, cid, tr, f"phishing victim: {len(fraud)} fraud tx; cards blocked", proposed)
            out.escalated = True
            return ReplyFacts("phishing_victim", tr.language, details=", ".join(blocked)), out
        self._set_status(ticket_id, "auto_resolved")
        return ReplyFacts("phishing_warning", tr.language, details=tr.signature.scam_domain or "посилання з SMS"), out

    async def _double_charge(self, ctx: ActionContext, cid: int, ticket_id: int, tr: TriageResult,
                             media: MediaFields | None, out: Reply) -> tuple[ReplyFacts, Reply]:
        """Verify the customer's own claim against the DB; screenshots are never proof."""
        pairs = impact.duplicate_pairs(self.db, tr.signature.merchant_id, ago(days=30), customer_id=cid)
        if pairs:
            p = pairs[-1]
            merchant = self.db.scalar("SELECT name FROM merchants WHERE id=?", (p["merchant_id"],))
            evidence = [f"tx #{p['first_id']} і #{p['dup_id']}: {p['amount']:.2f} грн, інтервал {p['gap_s']} с"]
            if ctx.level >= Level.SAFE_ACTION and "open_dispute" in policy.allowed_tools(ctx.level):
                self.tools.open_dispute(ctx, p["dup_id"], "duplicate transaction confirmed in data", evidence, ticket_id)
                out.actions.append(f"open_dispute tx#{p['dup_id']}")
            await self._escalate(ticket_id, cid, tr, f"duplicate confirmed: {evidence[0]} ({merchant})",
                                 {"action": "refund_transaction", "tx_id": p["dup_id"], "amount": p["amount"],
                                  "evidence": evidence})
            out.escalated = True
            return ReplyFacts("own_duplicate", tr.language, amount=p["amount"],
                              details=f"{merchant}, {local_time(p['first_at'])}"), out
        recent = self.tools.get_transactions(customer_id=cid, merchant_id=tr.signature.merchant_id, since_hours=72,
                                             limit=3)
        seen = ", ".join(f"{t['merchant']} {t['amount']:.0f} грн" for t in recent) or "операцій не знайдено"
        if media is not None and (media.amount or media.merchant):
            await self._escalate(ticket_id, cid, tr, f"screenshot claims double charge, DB has no duplicate ({seen})")
            out.escalated = True
            return ReplyFacts("screenshot_mismatch", tr.language, details=seen), out
        if set(tr.risk_flags) & {"vip_pressure", "unverified_claim", "angry"}:
            await self._escalate(ticket_id, cid, tr, f"claim not confirmed by data; flags={tr.risk_flags}")
            out.escalated = True
            return ReplyFacts("refund_unfounded", tr.language, details=seen), out
        self._set_status(ticket_id, "waiting_customer")
        return ReplyFacts("own_no_duplicate", tr.language, details=seen), out
