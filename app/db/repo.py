"""Thin SQLite access layer and time helpers."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
KYIV = ZoneInfo("Europe/Kyiv")
TS_FMT = "%Y-%m-%d %H:%M:%S"


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def ts(dt: datetime) -> str:
    """Serialize a datetime as a sortable UTC string."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime(TS_FMT)


def now_ts() -> str:
    return ts(utcnow())


def ago(**kwargs: float) -> str:
    return ts(utcnow() - timedelta(**kwargs))


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, TS_FMT).replace(tzinfo=timezone.utc)


def local_time(value: str | None) -> str:
    """Render a stored UTC timestamp as Kyiv HH:MM."""
    if not value:
        return "—"
    return parse_ts(value).astimezone(KYIV).strftime("%H:%M")


def money(amount: float | None) -> str:
    """1234.5 -> '1 234.50', 450.0 -> '450'."""
    if amount is None:
        return ""
    text = f"{amount:,.2f}".replace(",", " ")
    return text[:-3] if text.endswith(".00") else text


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class Database:
    """Single-connection SQLite wrapper (the bot runs in one event loop)."""

    def __init__(self, path: str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL" if path != ":memory:" else "PRAGMA journal_mode = MEMORY")

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    def has_schema(self) -> bool:
        row = self.q1("SELECT name FROM sqlite_master WHERE type='table' AND name='customers'")
        return row is not None

    def q(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, tuple(params)).fetchall()

    def q1(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, tuple(params)).fetchone()

    def scalar(self, sql: str, params: Iterable[Any] = ()) -> Any:
        row = self.q1(sql, params)
        return None if row is None else row[0]

    def ex(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, tuple(params))

    def insert(self, table: str, values: dict[str, Any]) -> int:
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        cur = self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(values.values()))
        return int(cur.lastrowid)

    def close(self) -> None:
        self.conn.close()
