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
        self._fallback_structured: dict[str, dict[str, Any]] = {}
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
            key = self._key(session_id)
            self.client.rpush(key, json.dumps(message, ensure_ascii=False))
            self.client.ltrim(key, -self.settings.max_history_messages, -1)
            # 每次写入都刷新 TTL，让活跃会话持续保留，不活跃会话在 7 天后自动过期。
            self.client.expire(key, self.settings.redis_ttl_seconds)
        except Exception:
            self.client = None
            self._append_fallback(session_id, message)

    def load_structured_memory(self, session_id: str) -> dict[str, Any]:
        if self.client is None:
            return dict(self._fallback_structured.get(session_id, self._empty_structured_memory()))
        try:
            raw = self.client.get(self._structured_key(session_id))
            if not raw:
                return self._empty_structured_memory()
            payload = json.loads(raw)
            return {
                "user_facts": dict(payload.get("user_facts", {})),
                "ticket_context": dict(payload.get("ticket_context", {})),
            }
        except Exception:
            self.client = None
            return dict(self._fallback_structured.get(session_id, self._empty_structured_memory()))

    def save_structured_memory(self, session_id: str, structured_memory: dict[str, Any]) -> None:
        payload = {
            "user_facts": dict(structured_memory.get("user_facts", {})),
            "ticket_context": dict(structured_memory.get("ticket_context", {})),
        }
        if self.client is None:
            self._fallback_structured[session_id] = payload
            return
        try:
            key = self._structured_key(session_id)
            self.client.set(key, json.dumps(payload, ensure_ascii=False))
            self.client.expire(key, self.settings.redis_ttl_seconds)
        except Exception:
            self.client = None
            self._fallback_structured[session_id] = payload

    def _append_fallback(self, session_id: str, message: dict[str, Any]) -> None:
        self._fallback.setdefault(session_id, []).append(message)
        self._fallback[session_id] = self._fallback[session_id][-self.settings.max_history_messages :]

    @staticmethod
    def _empty_structured_memory() -> dict[str, Any]:
        return {"user_facts": {}, "ticket_context": {}}

    @staticmethod
    def _key(session_id: str) -> str:
        return f"customer_service:session:{session_id}"

    @staticmethod
    def _structured_key(session_id: str) -> str:
        return f"customer_service:structured:{session_id}"