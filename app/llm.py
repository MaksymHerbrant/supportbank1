"""LLM adapter for Gemma 4 with two backends (Ollama / Google AI) and a strict-JSON fallback.

The adapter never raises to callers: failures return ``None`` so every caller
can degrade to deterministic logic and the bot keeps working.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
CIRCUIT_OPEN_S = 30.0  # after repeated failures, skip the LLM for a while (rule-based mode)


@dataclass
class ToolSpec:
    """A function the model may call."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str
    images: list[bytes] = field(default_factory=list)
    name: str | None = None


class LLMBackend(Protocol):
    async def chat(self, messages: list[Message], tools: list[ToolSpec] | None) -> LLMResponse: ...


def _schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Inline $defs so small models and both APIs get a flat JSON schema."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def extract_json(text: str) -> Any:
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    if not text:
        raise ValueError("empty reply")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in reply")
        candidate = text[start : end + 1]
    return json.loads(candidate)


class OllamaBackend:
    """Ollama via its OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings) -> None:
        from openai import AsyncOpenAI

        self.model = settings.ollama_model
        self.client = AsyncOpenAI(
            base_url=settings.ollama_url.rstrip("/") + "/v1", api_key="ollama",
            timeout=settings.llm_timeout_s, max_retries=0,
        )

    async def chat(self, messages: list[Message], tools: list[ToolSpec] | None) -> LLMResponse:
        payload: list[dict[str, Any]] = []
        for m in messages:
            if m.images:
                parts: list[dict[str, Any]] = [{"type": "text", "text": m.content}]
                for img in m.images:
                    b64 = base64.b64encode(img).decode()
                    parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                payload.append({"role": m.role, "content": parts})
            elif m.role == "tool":
                payload.append({"role": "tool", "content": m.content, "tool_call_id": m.name or "call"})
            else:
                payload.append({"role": m.role, "content": m.content})
        kwargs: dict[str, Any] = {"model": self.model, "messages": payload, "temperature": 0.1}
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        resp = await self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except ValueError:
                args = {}
            calls.append(ToolCall(name=tc.function.name, arguments=args))
        return LLMResponse(text=msg.content or "", tool_calls=calls)


class GoogleBackend:
    """Gemma 4 via the google-genai SDK."""

    def __init__(self, settings: Settings) -> None:
        from google import genai

        self.model = settings.google_model
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.timeout = settings.llm_timeout_s

    async def chat(self, messages: list[Message], tools: list[ToolSpec] | None) -> LLMResponse:
        from google.genai import types

        system = "\n".join(m.content for m in messages if m.role == "system")
        contents = []
        for m in messages:
            if m.role == "system":
                continue
            parts = [types.Part.from_text(text=m.content)]
            parts += [types.Part.from_bytes(data=img, mime_type="image/jpeg") for img in m.images]
            role = "model" if m.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=parts))
        if system and contents:
            # Gemma on the Gemini API does not accept system_instruction; prepend instead.
            contents[0].parts.insert(0, types.Part.from_text(text=system))
        config = types.GenerateContentConfig(temperature=0.1)
        if tools:
            config.tools = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
                for t in tools
            ])]
        resp = await asyncio.wait_for(
            self.client.aio.models.generate_content(model=self.model, contents=contents, config=config),
            timeout=self.timeout,
        )
        calls = [ToolCall(name=fc.name, arguments=dict(fc.args or {})) for fc in (resp.function_calls or [])]
        text = ""
        try:
            text = resp.text or ""
        except (ValueError, AttributeError):
            text = ""
        return LLMResponse(text=text, tool_calls=calls)


class LLMClient:
    """Robust facade: retries, timeouts, function calling with JSON fallback."""

    def __init__(self, backend: LLMBackend | None, retries: int = 2, timeout_s: float = 45.0) -> None:
        self.backend = backend
        self.retries = retries
        self.timeout_s = timeout_s
        self._down_until = 0.0

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient":
        backend: LLMBackend | None = None
        try:
            if settings.llm_backend == "ollama":
                backend = OllamaBackend(settings)
            elif settings.llm_backend == "google" and settings.gemini_api_key:
                backend = GoogleBackend(settings)
        except Exception:  # noqa: BLE001 - missing SDK / bad config must not crash the bot
            log.exception("LLM backend init failed; running in rule-based mode")
        return cls(backend, retries=settings.llm_retries, timeout_s=settings.llm_timeout_s)

    @property
    def available(self) -> bool:
        return self.backend is not None and time.monotonic() >= self._down_until

    async def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse | None:
        """Send a chat request; returns None after exhausting retries."""
        if not self.available:
            return None
        for attempt in range(self.retries + 1):
            try:
                return await asyncio.wait_for(self.backend.chat(messages, tools), timeout=self.timeout_s + 5)
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM call failed (attempt %s/%s): %s", attempt + 1, self.retries + 1, exc)
                if attempt < self.retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
        self._down_until = time.monotonic() + CIRCUIT_OPEN_S
        log.warning("LLM unavailable; rule-based mode for %ss", CIRCUIT_OPEN_S)
        return None

    async def text(self, system: str, user: str) -> str | None:
        resp = await self.chat([Message("system", system), Message("user", user)])
        return resp.text.strip() if resp and resp.text else None

    async def structured(
        self, system: str, user: str, schema: type[T], images: list[bytes] | None = None,
        tool_name: str = "submit_result",
    ) -> T | None:
        """Get a validated pydantic object.

        1) native function calling with a single tool whose parameters are the schema;
        2) fallback: ask for strict JSON and validate it ourselves.
        """
        if not self.available:
            return None
        params = _schema_for(schema)
        tool = ToolSpec(tool_name, f"Submit the final {schema.__name__} result.", params)
        messages = [Message("system", system), Message("user", user, images=list(images or []))]
        resp = await self.chat(messages, tools=[tool])
        if resp is not None:
            for call in resp.tool_calls:
                if call.name == tool_name:
                    try:
                        return schema.model_validate(call.arguments)
                    except ValidationError as exc:
                        log.info("Tool call failed validation, falling back to JSON: %s", exc)
            if resp.text:
                try:
                    return schema.model_validate(extract_json(resp.text))
                except (ValueError, ValidationError):
                    pass
        strict = (
            f"{user}\n\nReturn ONLY one JSON object (no prose, no markdown) matching this JSON schema:\n"
            f"{json.dumps(params, ensure_ascii=False)}"
        )
        resp = await self.chat([Message("system", system), Message("user", strict, images=list(images or []))])
        if resp is None or not resp.text:
            return None
        try:
            return schema.model_validate(extract_json(resp.text))
        except (ValueError, ValidationError) as exc:
            log.warning("LLM JSON fallback failed validation: %s", exc)
            return None
