"""Human handoff: while a live chat is open, customer messages bypass the AI and go to the operator group;
operators answer by replying to those messages. Every message is stored (masked) for the record."""
from __future__ import annotations

from app.agent import policy
from app.db.repo import Database, now_ts


class LiveChats:
    def __init__(self, db: Database) -> None:
        self.db = db

    def active(self, customer_id: int) -> dict | None:
        row = self.db.q1("SELECT * FROM live_chats WHERE customer_id=? AND status='open' ORDER BY id DESC LIMIT 1",
                         (customer_id,))
        return dict(row) if row else None

    def open(self, customer_id: int, operator_name: str) -> tuple[dict, bool]:
        """Open (or return the already open) chat. Returns (chat, created)."""
        existing = self.active(customer_id)
        if existing:
            return existing, False
        chat_id = self.db.insert("live_chats", {"customer_id": customer_id, "operator_name": operator_name,
                                                "started_at": now_ts()})
        self.db.ex("UPDATE escalations SET status='closed', decided_by=? WHERE customer_id=? AND status='open' "
                   "AND proposed_json IS NULL", (f"live chat: {operator_name}", customer_id))
        return dict(self.db.q1("SELECT * FROM live_chats WHERE id=?", (chat_id,))), True

    def close(self, customer_id: int) -> bool:
        cur = self.db.ex("UPDATE live_chats SET status='closed', ended_at=? WHERE customer_id=? AND status='open'",
                         (now_ts(), customer_id))
        return bool(cur.rowcount)

    def remember(self, group_message_id: int, customer_id: int, direction: str, text: str | None = None) -> None:
        self.db.ex("INSERT OR REPLACE INTO relay_messages (group_message_id, customer_id, direction, text, created_at) "
                   "VALUES (?,?,?,?,?)", (group_message_id, customer_id, direction, policy.mask_pan(text), now_ts()))

    def customer_of(self, group_message_id: int) -> int | None:
        cid = self.db.scalar("SELECT customer_id FROM relay_messages WHERE group_message_id=?", (group_message_id,))
        if cid is None:
            cid = self.db.scalar("SELECT customer_id FROM escalations WHERE operator_msg_id=?", (group_message_id,))
        return int(cid) if cid is not None else None
