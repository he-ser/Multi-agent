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
                    self._ensure_schema(cursor)
                    cursor.execute(
                        """
                        INSERT INTO conversation_turns (
                            session_id,
                            user_id,
                            intent,
                            assigned_agent,
                            product,
                            topic,
                            quality_score,
                            requires_human_review,
                            error_code,
                            order_id,
                            company_name,
                            contact_email,
                            team_size,
                            payload
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            payload["session_id"],
                            payload["user_id"],
                            payload.get("intent"),
                            payload.get("assigned_agent"),
                            payload.get("product"),
                            payload.get("topic"),
                            payload.get("quality_score"),
                            payload.get("requires_human_review"),
                            payload.get("error_code"),
                            payload.get("order_id"),
                            payload.get("company_name"),
                            payload.get("contact_email"),
                            payload.get("team_size"),
                            psycopg.types.json.Jsonb(payload),
                        ),
                    )
                conn.commit()
        except Exception:
            return

    def _ensure_schema(self, cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                intent TEXT,
                assigned_agent TEXT,
                product TEXT,
                topic TEXT,
                quality_score INTEGER,
                requires_human_review BOOLEAN,
                error_code TEXT,
                order_id TEXT,
                company_name TEXT,
                contact_email TEXT,
                team_size TEXT,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        # 兼容已存在的旧表，增量补齐结构化字段。
        for ddl in [
            "ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS product TEXT",
            "ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS topic TEXT",
            "ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS quality_score INTEGER",
            "ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS error_code TEXT",
            "ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS order_id TEXT",
            "ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS company_name TEXT",
            "ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS contact_email TEXT",
            "ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS team_size TEXT",
        ]:
            cursor.execute(ddl)
        for ddl in [
            "CREATE INDEX IF NOT EXISTS idx_conversation_turns_session_id ON conversation_turns(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_turns_intent ON conversation_turns(intent)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_turns_product ON conversation_turns(product)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_turns_topic ON conversation_turns(topic)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_turns_error_code ON conversation_turns(error_code)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_turns_created_at ON conversation_turns(created_at)",
        ]:
            cursor.execute(ddl)

    def cleanup_expired_turns(self) -> int:
        if psycopg is None:
            return 0
        try:
            with psycopg.connect(self.settings.postgres_dsn) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        DELETE FROM conversation_turns
                        WHERE created_at < NOW() - (%s * INTERVAL '1 day')
                        """,
                        (self.settings.postgres_retention_days,),
                    )
                    deleted_count = cursor.rowcount or 0
                conn.commit()
            return deleted_count
        except Exception:
            return 0

    def close(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=1)