"""FAQ RAG answers only from fragments; otherwise escalates."""
from __future__ import annotations

from app.agent.faq_rag import FaqRAG, load_chunks
from app.llm import LLMClient, LLMResponse
from app.vector.store import VectorStore
from app.embed import NullEmbedder
from tests.conftest import ConceptEmbedder, FakeBackend


def test_faq_has_enough_chunks() -> None:
    assert 15 <= len(load_chunks()) <= 20


async def test_relevant_question_answered_from_fragment(db) -> None:  # noqa: ANN001
    rag = FaqRAG(LLMClient(None), VectorStore(db, NullEmbedder()))
    ans = await rag.answer("Який ліміт на зняття готівки?")
    assert not ans.escalate and "30 000" in ans.text


async def test_irrelevant_question_escalates_instead_of_inventing(db) -> None:  # noqa: ANN001
    rag = FaqRAG(LLMClient(None), VectorStore(db, NullEmbedder()))
    ans = await rag.answer("Як вирощувати помідори на балконі?")
    assert ans.escalate and ans.text is None


async def test_model_saying_no_answer_escalates(db) -> None:  # noqa: ANN001
    llm = LLMClient(FakeBackend(lambda m, t: LLMResponse(text="NO_ANSWER")))
    rag = FaqRAG(llm, VectorStore(db, ConceptEmbedder()))
    await rag.index()
    ans = await rag.answer("Який ліміт на зняття готівки?")
    assert ans.escalate and ans.text is None


async def test_vector_retrieval(db) -> None:  # noqa: ANN001
    rag = FaqRAG(LLMClient(None), VectorStore(db, ConceptEmbedder()))
    assert await rag.index() == len(rag.chunks)
    hits = await rag.retrieve("ліміт готівки на день")
    assert hits and "готівк" in hits[0].question.lower()


async def test_pipeline_asks_then_escalates_unknown_faq(container) -> None:  # noqa: ANN001
    first = await container.pipeline.handle(91, "Як вирощувати помідори на балконі?")
    assert not first.escalated and "?" in first.text  # the agent tries to clarify first, never invents
    replies = [await container.pipeline.handle(91, "Як вирощувати помідори на балконі?") for _ in range(3)]
    assert any(r.escalated for r in replies)
