from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None

from customer_service.config import get_settings


class AsyncConversationRepository:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.queue: Queue[dict[str, Any]] = Queue()
        self.stop_event = Event()
        self.worker = Thread(target=self._run, daemon=True)
        self.worker.start()

    def save_turn(self, payload: dict[str, Any]) -> None:
        self.queue.put(payload)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                payload = self.queue.get(timeout=0.2)
            except Empty:
                continue
            self._persist(payload)
            self.queue.task_done()

    def _persist(self, payload: dict[str, Any]) -> None:
        if psycopg is None:
            return
        try:
            with psycopg.connect(self.settings.postgres_dsn) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS conversation_turns (
                            id SERIAL PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            intent TEXT,
                            assigned_agent TEXT,
                            requires_human_review BOOLEAN,
                            payload JSONB NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                        """
                    )
                    cursor.execute(
                        """
                        INSERT INTO conversation_turns
                        (session_id, user_id, intent, assigned_agent, requires_human_review, payload)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            payload["session_id"],
                            payload["user_id"],
                            payload.get("intent"),
                            payload.get("assigned_agent"),
                            payload.get("requires_human_review"),
                            psycopg.types.json.Jsonb(payload),
                        ),
                    )
                conn.commit()
        except Exception:
            return

    def close(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=1)
