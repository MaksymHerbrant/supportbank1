"""Media: card masking and no persistence of originals."""
from __future__ import annotations

import pytest

from app.agent import policy
from app.agent.media import MAX_VIDEO_BYTES, MediaAnalyzer, MediaError, extract_frames
from app.agent.triage import MediaFields
from app.llm import LLMClient, LLMResponse, ToolCall
from tests.conftest import FakeBackend


def test_card_number_masked_to_last4() -> None:
    assert MediaFields(card_last4="4149 4993 1234 5678").card_last4 == "5678"
    assert policy.mask_pan("картка 4149499312345678 заблокована") == "картка **** 5678 заблокована"
    assert policy.mask_pan("4149-4993-1234-5678") == "**** 5678"
    assert policy.mask_pan("сума 450 грн") == "сума 450 грн"


async def test_vision_output_is_masked() -> None:
    args = {"error_text": "Card 4149 4993 1234 5678 declined", "card_last4": "4149499312345678", "amount": 450}
    llm = LLMClient(FakeBackend(lambda m, t: LLMResponse(tool_calls=[ToolCall("submit_media_fields", args)])))
    fields = await MediaAnalyzer(llm).from_images([b"\x89PNG fake"])
    assert fields is not None
    assert fields.card_last4 == "5678"
    assert "4149" not in (fields.error_text or "")
    assert "**** 5678" in fields.error_text


async def test_only_extracted_fields_are_stored(container) -> None:  # noqa: ANN001
    media = MediaFields(error_text="Помилка E-PAY-502", app_version="5.4.0", card_last4="1234")
    reply = await container.pipeline.handle(59, "", media)
    row = container.db.q1("SELECT text, media_json FROM tickets WHERE id=?", (reply.ticket_id,))
    stored = MediaFields.model_validate_json(row["media_json"])
    assert stored == media
    assert "PNG" not in (row["media_json"] or "")
    tables = {r[0] for r in container.db.q("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in tables:
        cols = {r[1] for r in container.db.q(f"PRAGMA table_info({table})")}
        assert not cols & {"image", "photo", "video", "blob_data"}


async def test_text_card_number_masked_in_ticket(container) -> None:  # noqa: ANN001
    reply = await container.pipeline.handle(91, "Моя картка 4149 4993 1234 5678 загубилась")
    text = container.db.scalar("SELECT text FROM tickets WHERE id=?", (reply.ticket_id,))
    assert "4149" not in text and "5678" in text


async def test_video_size_limit() -> None:
    with pytest.raises(MediaError):
        await extract_frames(b"0" * (MAX_VIDEO_BYTES + 1))


async def test_no_llm_returns_none() -> None:
    assert await MediaAnalyzer(LLMClient(None)).from_images([b"x"]) is None
