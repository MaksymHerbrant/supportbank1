"""LLM adapter: function calling, strict-JSON fallback, failures never raise."""
from __future__ import annotations

from pydantic import BaseModel

from app.llm import LLMClient, LLMResponse, ToolCall, extract_json
from tests.conftest import FakeBackend


class Out(BaseModel):
    category: str
    level: int


async def test_native_tool_call() -> None:
    llm = LLMClient(FakeBackend(lambda m, t: LLMResponse(tool_calls=[ToolCall("submit_result", {"category": "faq", "level": 1})])))
    assert await llm.structured("s", "u", Out) == Out(category="faq", level=1)


async def test_invalid_tool_call_falls_back_to_strict_json() -> None:
    calls = []

    def respond(messages, tools):  # noqa: ANN001, ANN202
        calls.append(tools)
        if tools:
            return LLMResponse(tool_calls=[ToolCall("submit_result", {"category": "faq"})])  # missing level
        return LLMResponse(text='```json\n{"category": "atm_capture", "level": 2}\n```')

    llm = LLMClient(FakeBackend(respond))
    assert await llm.structured("s", "u", Out) == Out(category="atm_capture", level=2)
    assert calls[-1] is None  # second call asked for plain JSON


async def test_garbage_returns_none() -> None:
    llm = LLMClient(FakeBackend(lambda m, t: LLMResponse(text="I cannot help with that")))
    assert await llm.structured("s", "u", Out) is None


async def test_backend_exceptions_are_swallowed_and_circuit_opens() -> None:
    def boom(m, t):  # noqa: ANN001, ANN202
        raise TimeoutError("down")

    backend = FakeBackend(boom)
    llm = LLMClient(backend, retries=1)
    assert await llm.text("s", "u") is None
    assert not llm.available
    calls = len(backend.calls)
    assert await llm.text("s", "u") is None
    assert len(backend.calls) == calls  # short-circuited


def test_extract_json() -> None:
    assert extract_json('Here: {"a": 1} done') == {"a": 1}
