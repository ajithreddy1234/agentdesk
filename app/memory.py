"""Long-term user memory (Redis hash per customer, in-process dict fallback).

Short-term memory is the LangGraph thread state (messages + rolling summary); this store keeps
facts across conversations, such as a preferred refund method.
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache

from .config import settings

ALLOWED_KEYS = {"preferred_refund_method", "preferred_language", "delivery_instructions", "dietary_preference", "note"}
MAX_VALUE_LEN = 200


class MemoryStore:
    def __init__(self, redis_url: str | None = None):
        self._local: dict[int, dict[str, str]] = {}
        self._lock = threading.Lock()
        self.r = None
        if redis_url:
            import redis

            self.r = redis.Redis.from_url(redis_url, decode_responses=True)

    def get(self, customer_id: int) -> dict[str, str]:
        if self.r:
            return self.r.hgetall(f"agentdesk:mem:{customer_id}")
        with self._lock:
            return dict(self._local.get(customer_id, {}))

    def set(self, customer_id: int, key: str, value: str) -> dict[str, str]:
        if key not in ALLOWED_KEYS:
            raise ValueError(f"key must be one of {sorted(ALLOWED_KEYS)}")
        value = value.strip()[:MAX_VALUE_LEN]
        if self.r:
            self.r.hset(f"agentdesk:mem:{customer_id}", key, value)
            self.r.expire(f"agentdesk:mem:{customer_id}", 60 * 60 * 24 * 180)
        else:
            with self._lock:
                self._local.setdefault(customer_id, {})[key] = value
        return self.get(customer_id)

    def render(self, customer_id: int) -> str:
        m = self.get(customer_id)
        return json.dumps(m) if m else "{}"


@lru_cache(maxsize=1)
def get_memory() -> MemoryStore:
    return MemoryStore(settings.redis_url)
