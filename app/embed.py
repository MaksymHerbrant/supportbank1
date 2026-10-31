"""Embedding adapter (Ollama embeddinggemma -> nomic-embed-text -> unavailable).

When no model is reachable the client returns ``None`` and the system keeps
working on exact signatures only (graceful degradation).
"""
from __future__ import annotations

import logging
import time
from typing import Protocol

import httpx
import numpy as np

from app.config import Settings

log = logging.getLogger(__name__)
RETRY_AFTER_S = 60.0


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[np.ndarray] | None: ...


class EmbedClient:
    """Calls Ollama /api/embed, falling back between models and caching failures."""

    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.embed_backend == "ollama"
        self.url = settings.ollama_url.rstrip("/") + "/api/embed"
        self.models = [m for m in (settings.embed_model, settings.embed_fallback_model) if m]
        self._down_until = 0.0
        self._active: str | None = None

    async def _call(self, model: str, texts: list[str]) -> list[np.ndarray]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(self.url, json={"model": model, "input": texts})
            resp.raise_for_status()
            data = resp.json()["embeddings"]
        return [np.asarray(v, dtype=np.float32) for v in data]

    async def embed(self, texts: list[str]) -> list[np.ndarray] | None:
        if not self.enabled or not texts or time.monotonic() < self._down_until:
            return None
        order = [self._active] + [m for m in self.models if m != self._active] if self._active else self.models
        for model in order:
            try:
                vectors = await self._call(model, texts)
                if self._active != model:
                    log.info("Embeddings via %s", model)
                self._active = model
                return vectors
            except Exception as exc:  # noqa: BLE001
                log.warning("Embedding model %s unavailable: %s", model, exc)
        self._down_until = time.monotonic() + RETRY_AFTER_S
        log.warning("No embedding model reachable; exact-signature mode for %ss", RETRY_AFTER_S)
        return None


class NullEmbedder:
    """Embeddings disabled."""

    async def embed(self, texts: list[str]) -> list[np.ndarray] | None:
        return None
