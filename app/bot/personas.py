"""Demo personas a real Telegram user can impersonate. Labels show only the name: the bot must work out
the problem itself from the conversation and the data."""
from __future__ import annotations

from app.db.repo import Database

# slug for /demo_login -> customers.demo_tag
PERSONAS: dict[str, str] = {
    "loyal": "loyal",
    "refunder": "frequent_refunder",
    "newphone": "phone_changed",
    "credit_overdue": "credit_overdue",
    "credit_ok": "credit_ok",
    "deferral": "credit_deferral",
    "fuel": "fuel_late_affected",
    "fuel_clean": "fuel_late_clean",
    "phishing": "phish_reporter_3",
    "atm": "atm_reporter_3",
    "app": "app_reporter_3",
    "screenshot": "fake_screenshot",
    "social": "social_eng",
    "vip": "vip_demand",
    "faq": "faq_user",
}
DEFAULT_PERSONA = "fuel"


def customer_by_slug(db: Database, slug: str) -> int | None:
    tag = PERSONAS.get(slug.strip().lower())
    cid = db.scalar("SELECT id FROM customers WHERE demo_tag=?", (tag,)) if tag else None
    return int(cid) if cid else None


def bind(db: Database, chat_id: int, customer_id: int) -> str:
    db.ex("UPDATE customers SET tg_chat_id=NULL WHERE tg_chat_id=?", (chat_id,))
    db.ex("UPDATE customers SET tg_chat_id=? WHERE id=?", (chat_id, customer_id))
    return db.scalar("SELECT full_name FROM customers WHERE id=?", (customer_id,)) or ""


def listing(db: Database) -> list[tuple[str, int, str]]:
    """(slug, customer_id, name) for every persona present in the DB."""
    out = []
    for slug in PERSONAS:
        cid = customer_by_slug(db, slug)
        if cid:
            out.append((slug, cid, db.scalar("SELECT full_name FROM customers WHERE id=?", (cid,))))
    return out
