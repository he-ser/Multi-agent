from __future__ import annotations

import json
from typing import Any

try:
    import redis
except Exception:  # pragma: no cover
    redis = None

from customer_service.config import get_settings


class RedisConversationMemory:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._fallback: dict[str, list[dict[str, Any]]] = {}
        self.client = self._build_client()

    def _build_client(self):
        if redis is None:
            return None
        try:
            client = redis.from_url(self.settings.redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception:
            return None

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        if self.client is None:
            return list(self._fallback.get(session_id, []))
        try:
            raw_items = self.client.lrange(self._key(session_id), 0, self.settings.max_history_messages - 1)
            return [json.loads(item) for item in raw_items]
        except Exception:
            self.client = None
            return list(self._fallback.get(session_id, []))

    def append_message(self, session_id: str, message: dict[str, Any]) -> None:
        if self.client is None:
            self._append_fallback(session_id, message)
            return
        try:
            self.client.rpush(self._key(session_id), json.dumps(message, ensure_ascii=False))
            self.client.ltrim(self._key(session_id), -self.settings.max_history_messages, -1)
        except Exception:
            self.client = None
            self._append_fallback(session_id, message)

    def _append_fallback(self, session_id: str, message: dict[str, Any]) -> None:
        self._fallback.setdefault(session_id, []).append(message)
        self._fallback[session_id] = self._fallback[session_id][-self.settings.max_history_messages :]

    @staticmethod
    def _key(session_id: str) -> str:
        return f"customer_service:session:{session_id}"
