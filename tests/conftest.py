"""Shared fixtures: seeded in-memory DB, fake LLM backend and a deterministic concept embedder."""
from __future__ import annotations

import hashlib
import re
from typing import Callable

import numpy as np
import pytest

from app.config import Settings
from app.container import Container, build
from app.db.repo import Database
from app.db.seed import seed
from app.embed import NullEmbedder
from app.llm import LLMClient, LLMResponse, Message, ToolSpec
from app.notify import RecordingNotifier

CONCEPTS: dict[str, list[str]] = {
    "dup": ["двічі", "подвійн", "дубл", "два рази", "двa рази", "twice", "double", "повторно"],
    "fuel": ["заправ", "азс", "fuel", "пальн", "бензин"],
    "money": ["гроші", "грош", "кошт", "списа", "зняли", "оплат"],
    "atm": ["банкомат", "atm"],
    "card": ["картк", "card"],
    "app": ["застосун", "додат", "app", "оновлен"],
    "error": ["помилк", "error", "не прох"],
    "sms": ["sms", "смс", "посилан", "link"],
    "limit": ["ліміт", "limit"],
    "cash": ["готівк", "cash"],
}


class ConceptEmbedder:
    """Deterministic 'semantic' embedder: paraphrases share concepts, other words add small noise."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        self.calls += 1
        out = []
        for text in texts:
            low = text.lower()
            vec = np.zeros(len(CONCEPTS) + 16, dtype=np.float32)
            for i, words in enumerate(CONCEPTS.values()):
                vec[i] = sum(low.count(w) > 0 for w in words)
            for word in re.findall(r"\w+", low):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16) % 16
                vec[len(CONCEPTS) + h] += 0.08
            out.append(vec)
        return out


class FakeBackend:
    """Scripted LLM backend: a function decides the response for each call."""

    def __init__(self, fn: Callable[[list[Message], list[ToolSpec] | None], LLMResponse]) -> None:
        self.fn = fn
        self.calls: list[tuple[list[Message], list[ToolSpec] | None]] = []

    async def chat(self, messages: list[Message], tools: list[ToolSpec] | None) -> LLMResponse:
        self.calls.append((messages, tools))
        return self.fn(messages, tools)


def settings(**overrides) -> Settings:  # noqa: ANN003
    base = dict(llm_backend="none", embed_backend="none", similarity_threshold=0.8, incident_threshold=3,
                incident_window_min=60)
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def db() -> Database:
    database = Database(":memory:")
    seed(database)
    yield database
    database.close()


@pytest.fixture
def notifier() -> RecordingNotifier:
    return RecordingNotifier()


def make_container(db: Database, notifier: RecordingNotifier, llm: LLMClient | None = None,
                   embedder=None, **overrides) -> Container:  # noqa: ANN001, ANN003
    return build(settings(**overrides), db, notifier, llm=llm or LLMClient(None),
                 embedder=embedder or NullEmbedder())


@pytest.fixture
def container(db: Database, notifier: RecordingNotifier) -> Container:
    """Rule-based mode: no LLM, no embeddings."""
    return make_container(db, notifier)


@pytest.fixture
def vec_container(db: Database, notifier: RecordingNotifier) -> Container:
    """No LLM, but semantic embeddings available."""
    return make_container(db, notifier, embedder=ConceptEmbedder())


async def run_double_charge(c: Container) -> int:
    """Three Fuel+ complaints from affected customers -> incident id."""
    for cid, text in [(1, "Мене двічі списали на АЗС Fuel+"), (2, "Fuel+ зняли гроші два рази"),
                      (3, "Подвійне списання fuel plus")]:
        reply = await c.pipeline.handle(cid, text)
    assert reply.new_incident_id is not None
    return reply.new_incident_id
