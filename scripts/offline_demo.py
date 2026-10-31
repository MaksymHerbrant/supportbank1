"""Run every /simulate scenario in the console, no Telegram needed.

    python -m scripts.offline_demo            # uses LLM/embeddings from .env if reachable
    python -m scripts.offline_demo --rules    # force rule-based mode (no LLM, no embeddings)
"""
from __future__ import annotations

import asyncio
import logging
import re
import sys

from app import stats
from app.bot import simulate
from app.bot.cards import render_incident
from app.config import get_settings
from app.container import build
from app.db.repo import Database
from app.db.seed import seed
from app.embed import NullEmbedder
from app.llm import LLMClient
from app.notify import RecordingNotifier


def plain(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    db = Database(":memory:")
    seed(db)
    rules = "--rules" in sys.argv
    c = build(get_settings(), db, RecordingNotifier(),
              llm=LLMClient(None) if rules else None, embedder=NullEmbedder() if rules else None)
    await c.faq.index()

    async def say(text: str) -> None:
        print(plain(text), end="\n\n")

    async def photo(path, caption: str) -> None:  # noqa: ANN001
        print(f"[фото {path.name}] {caption}")

    for name in ["double_charge", "late_complaint", "phishing", "atm", "app_bug", "risky"]:
        await simulate.run(c, name, say, photo, delay=0)
    await c.incidents.drain()
    plan_id = db.scalar("SELECT id FROM plans WHERE status='pending' ORDER BY id LIMIT 1")
    print("=" * 70)
    print(plain(render_incident(db, 1)[0]))
    print("=" * 70, f"\nОператор підтверджує план #{plan_id}…")
    print(await c.executor.approve_plan(plan_id, "demo-operator"))
    print(plain(stats.render(stats.compute(db))))


if __name__ == "__main__":
    asyncio.run(main())
