from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./agentdesk.db")
    redis_url: str | None = os.getenv("REDIS_URL")  # None -> in-process memory store
    checkpoint_url: str | None = os.getenv("CHECKPOINT_URL")  # postgres URL for LangGraph checkpoints
    vector_backend: str = os.getenv("VECTOR_BACKEND", "memory")  # "pgvector" in production
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")  # or "hash"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    llm_api_key: str | None = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    judge_model: str = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")
    service_base_url: str | None = os.getenv("SERVICE_BASE_URL")  # None -> call order service in-process
    auto_refund_limit: float = float(os.getenv("AUTO_REFUND_LIMIT", "500"))
    summarize_after_messages: int = int(os.getenv("SUMMARIZE_AFTER_MESSAGES", "16"))
    keep_recent_messages: int = int(os.getenv("KEEP_RECENT_MESSAGES", "6"))
    max_tool_rounds: int = int(os.getenv("MAX_TOOL_ROUNDS", "8"))
    # Real clock by default. Set AGENTDESK_NOW (ISO time) only to freeze time in tests or demos.
    now_override: str | None = os.getenv("AGENTDESK_NOW") or None

    def now(self) -> datetime:
        return datetime.fromisoformat(self.now_override) if self.now_override else datetime.now()


settings = Settings()
