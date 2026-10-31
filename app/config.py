"""Application settings loaded from environment / .env."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DEMO_ASSETS_DIR = ROOT_DIR / "demo_assets"


class Settings(BaseModel):
    """Typed runtime configuration."""

    bot_token: str = ""
    operator_chat_id: int = 0
    dev_chat_id: int = 0

    llm_backend: str = "ollama"  # ollama | google | none
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e4b"
    google_model: str = "gemma-4-26b-a4b-it"
    gemini_api_key: str = ""
    llm_timeout_s: float = 45.0
    llm_retries: int = 2

    embed_backend: str = "ollama"  # ollama | none
    embed_model: str = "embeddinggemma"
    embed_fallback_model: str = "nomic-embed-text"

    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0)
    incident_threshold: int = Field(3, ge=1)
    incident_window_min: int = Field(60, ge=1)

    db_path: str = str(DATA_DIR / "demobank.db")
    log_level: str = "INFO"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""
    load_dotenv(ROOT_DIR / ".env")
    raw = {
        "bot_token": _env("BOT_TOKEN", ""),
        "operator_chat_id": _env("OPERATOR_CHAT_ID", "0"),
        "dev_chat_id": _env("DEV_CHAT_ID", "0"),
        "llm_backend": _env("LLM_BACKEND", "ollama"),
        "ollama_url": _env("OLLAMA_URL", "http://localhost:11434"),
        "ollama_model": _env("OLLAMA_MODEL", "gemma4:e4b"),
        "google_model": _env("GOOGLE_MODEL", "gemma-4-26b-a4b-it"),
        "gemini_api_key": _env("GEMINI_API_KEY", ""),
        "llm_timeout_s": _env("LLM_TIMEOUT_S", "45"),
        "llm_retries": _env("LLM_RETRIES", "2"),
        "embed_backend": _env("EMBED_BACKEND", "ollama"),
        "embed_model": _env("EMBED_MODEL", "embeddinggemma"),
        "embed_fallback_model": _env("EMBED_FALLBACK_MODEL", "nomic-embed-text"),
        "similarity_threshold": _env("SIMILARITY_THRESHOLD", "0.8"),
        "incident_threshold": _env("INCIDENT_THRESHOLD", "3"),
        "incident_window_min": _env("INCIDENT_WINDOW_MIN", "60"),
        "db_path": _env("DB_PATH", str(DATA_DIR / "demobank.db")),
        "log_level": _env("LOG_LEVEL", "INFO"),
    }
    settings = Settings.model_validate(raw)
    if not Path(settings.db_path).is_absolute():
        settings.db_path = str(ROOT_DIR / settings.db_path)
    return settings
