"""Composition root: builds every component from settings (or test doubles)."""
from __future__ import annotations

from dataclasses import dataclass

from app.actions.executor import Executor
from app.agent.faq_rag import FaqRAG
from app.agent.media import MediaAnalyzer
from app.agent.pipeline import SupportPipeline
from app.agent.responder import Responder
from app.agent.tools import BankTools
from app.agent.triage import Triage
from app.config import Settings
from app.db.repo import Database
from app.embed import EmbedClient, Embedder
from app.incidents.service import IncidentService
from app.llm import LLMClient
from app.notify import Notifier
from app.vector.store import VectorStore


@dataclass
class Container:
    settings: Settings
    db: Database
    llm: LLMClient
    store: VectorStore
    tools: BankTools
    faq: FaqRAG
    media: MediaAnalyzer
    responder: Responder
    incidents: IncidentService
    executor: Executor
    pipeline: SupportPipeline
    notifier: "LateNotifier"


class LateNotifier:
    """Lets the bot attach its Telegram notifier after the container is built."""

    def __init__(self, target: Notifier) -> None:
        self.target = target

    async def send_customer(self, customer_id: int, text: str) -> bool:
        return await self.target.send_customer(customer_id, text)

    async def incident_changed(self, incident_id: int) -> None:
        await self.target.incident_changed(incident_id)

    async def escalation_created(self, escalation_id: int) -> None:
        await self.target.escalation_created(escalation_id)

    async def dev_ticket(self, bug_ticket_id: int) -> None:
        await self.target.dev_ticket(bug_ticket_id)


def build(settings: Settings, db: Database, notifier: Notifier, llm: LLMClient | None = None,
          embedder: Embedder | None = None) -> Container:
    llm = llm or LLMClient.from_settings(settings)
    store = VectorStore(db, embedder or EmbedClient(settings))
    late = LateNotifier(notifier)
    tools = BankTools(db, late)
    responder = Responder(llm)
    faq = FaqRAG(llm, store)
    incidents = IncidentService(db, llm, store, tools, late, responder, settings)
    pipeline = SupportPipeline(db, Triage(db, llm), tools, faq, responder, incidents)
    return Container(settings, db, llm, store, tools, faq, MediaAnalyzer(llm), responder, incidents,
                     Executor(db, tools), pipeline, late)
