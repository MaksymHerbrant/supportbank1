"""Server-less vector store: embeddings in SQLite, cosine similarity with numpy.

Collections: ``complaints``, ``incidents``, ``faq``. Vector similarity is only
used to group complaints and find candidates; it never decides who is affected.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.db.repo import Database, dumps, loads
from app.embed import Embedder

log = logging.getLogger(__name__)
COLLECTIONS = ("complaints", "incidents", "faq")


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class VectorStore:
    def __init__(self, db: Database, embedder: Embedder) -> None:
        self.db = db
        self.embedder = embedder

    async def add(self, collection: str, item_id: str, text: str, metadata: dict[str, Any] | None = None) -> bool:
        """Embed and upsert; returns False if embeddings are unavailable (text is still stored)."""
        assert collection in COLLECTIONS, collection
        vectors = await self.embedder.embed([text])
        blob = _normalize(vectors[0]).astype(np.float32).tobytes() if vectors else None
        self.db.ex(
            "INSERT INTO vectors (collection, id, text, metadata_json, embedding) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(collection, id) DO UPDATE SET text=excluded.text, "
            "metadata_json=excluded.metadata_json, embedding=COALESCE(excluded.embedding, vectors.embedding)",
            (collection, item_id, text, dumps(metadata or {}), blob),
        )
        return blob is not None

    async def add_many(self, collection: str, items: list[tuple[str, str, dict[str, Any]]]) -> int:
        vectors = await self.embedder.embed([text for _, text, _ in items]) if items else None
        for idx, (item_id, text, meta) in enumerate(items):
            blob = _normalize(vectors[idx]).astype(np.float32).tobytes() if vectors else None
            self.db.ex(
                "INSERT OR REPLACE INTO vectors (collection, id, text, metadata_json, embedding) VALUES (?, ?, ?, ?, ?)",
                (collection, item_id, text, dumps(meta), blob),
            )
        return len(items) if vectors else 0

    def update_metadata(self, collection: str, item_id: str, **changes: Any) -> None:
        row = self.db.q1("SELECT metadata_json FROM vectors WHERE collection=? AND id=?", (collection, item_id))
        if row is None:
            return
        meta = loads(row["metadata_json"], {})
        meta.update(changes)
        self.db.ex("UPDATE vectors SET metadata_json=? WHERE collection=? AND id=?", (dumps(meta), collection, item_id))

    async def search(
        self, collection: str, text: str, k: int = 5, filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Top-k by cosine similarity; empty list when embeddings are unavailable."""
        rows = self.db.q(
            "SELECT id, metadata_json, embedding FROM vectors WHERE collection=? AND embedding IS NOT NULL",
            (collection,),
        )
        candidates = []
        for row in rows:
            meta = loads(row["metadata_json"], {})
            if filter and any(meta.get(key) != value for key, value in filter.items()):
                continue
            candidates.append((row["id"], meta, np.frombuffer(row["embedding"], dtype=np.float32)))
        if not candidates:
            return []
        vectors = await self.embedder.embed([text])
        if not vectors:
            return []
        query = _normalize(vectors[0]).astype(np.float32)
        dims = {c[2].shape[0] for c in candidates}
        candidates = [c for c in candidates if c[2].shape[0] == query.shape[0]]
        if not candidates:
            log.warning("Vector dims mismatch (store=%s, query=%s); re-index needed", dims, query.shape[0])
            return []
        matrix = np.stack([c[2] for c in candidates])
        scores = matrix @ query
        order = np.argsort(-scores)[:k]
        return [(candidates[i][0], float(scores[i]), candidates[i][1]) for i in order]
