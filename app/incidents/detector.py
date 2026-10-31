"""Hybrid complaint clustering: exact signature first, then embedding similarity.

N+ similar complaints from distinct customers within T minutes -> incident.
"""
from __future__ import annotations

import logging
from collections import Counter

from app.agent.triage import INCIDENT_CATEGORIES, Signature
from app.db.repo import Database, ago, loads
from app.vector.store import VectorStore

log = logging.getLogger(__name__)


class IncidentDetector:
    def __init__(self, db: Database, store: VectorStore, threshold: int, window_min: int, similarity: float) -> None:
        self.db = db
        self.store = store
        self.threshold = threshold
        self.window_min = window_min
        self.similarity = similarity

    async def index_complaint(self, ticket_id: int, text: str, category: str) -> None:
        await self.store.add("complaints", str(ticket_id), text, {"category": category})

    async def find_cluster(self, ticket_id: int) -> list[int] | None:
        """Return ticket ids forming a new incident with ``ticket_id``, or None below threshold."""
        t = self.db.q1("SELECT * FROM tickets WHERE id=?", (ticket_id,))
        if t is None or t["category"] not in INCIDENT_CATEGORIES or t["incident_id"] is not None:
            return None
        sig = Signature.model_validate(loads(t["signature_json"], {}))
        key = sig.key()
        others = self.db.q(
            "SELECT id, customer_id, signature_json FROM tickets WHERE category=? AND incident_id IS NULL "
            "AND created_at >= ? AND id != ?", (t["category"], ago(minutes=self.window_min), ticket_id))
        if not others:
            return None
        sims: dict[int, float] = {}
        if any(Signature.model_validate(loads(o["signature_json"], {})).key() != key or not key for o in others):
            hits = await self.store.search("complaints", t["text"], k=100, filter={"category": t["category"]})
            sims = {int(i): s for i, s, _ in hits}
        cluster = {ticket_id: t["customer_id"]}
        for o in others:
            o_key = Signature.model_validate(loads(o["signature_json"], {})).key()
            if key and o_key:
                if key == o_key:
                    cluster[o["id"]] = o["customer_id"]
                continue  # both known and different -> different root cause
            if sims.get(o["id"], 0.0) >= self.similarity:
                cluster[o["id"]] = o["customer_id"]
        distinct_customers = len(set(cluster.values()))
        log.info("detector: ticket %s cluster=%s customers=%s threshold=%s", ticket_id, sorted(cluster),
                 distinct_customers, self.threshold)
        if distinct_customers < self.threshold:
            return None
        return sorted(cluster)

    def merged_signature(self, ticket_ids: list[int]) -> Signature:
        """Majority vote per field across the cluster."""
        sigs = [Signature.model_validate(loads(r["signature_json"], {}))
                for r in self.db.q(f"SELECT signature_json FROM tickets WHERE id IN ({','.join('?' * len(ticket_ids))})",
                                   ticket_ids)]
        merged: dict = {"category": sigs[0].category}
        for field in ("merchant_id", "atm_id", "app_version", "error_code", "scam_domain", "timeframe"):
            values = [getattr(s, field) for s in sigs if getattr(s, field)]
            merged[field] = Counter(values).most_common(1)[0][0] if values else None
        return Signature.model_validate(merged)
