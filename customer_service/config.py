from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    openai_api_key: str | None
    openai_base_url: str | None
    openai_model: str
    embedding_model: str
    langsmith_tracing: bool
    langsmith_api_key: str | None
    langsmith_project: str
    langsmith_endpoint: str
    redis_url: str
    redis_ttl_days: int
    postgres_dsn: str
    postgres_retention_days: int
    chroma_persist_dir: Path
    chroma_collection: str
    chroma_shared_collection: str
    max_history_messages: int
    top_k_retrieval: int
    quality_threshold: int
    rewrite_enabled: bool
    workflow_name: str
    workflow_diagram_path: Path

    @property
    def redis_ttl_seconds(self) -> int:
        return self.redis_ttl_days * 24 * 60 * 60


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("ALIYUN_API_KEY") or os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("ALIYUN_API_URL") or os.getenv("OPENAI_BASE_URL"),
        openai_model=os.getenv("ALIYUN_MODEL") or os.getenv("OPENAI_MODEL", "deepseek-r1-0528"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        langsmith_tracing=_to_bool(os.getenv("LANGSMITH_TRACING")),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "multi-agent-customer-service"),
        langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        redis_ttl_days=int(os.getenv("REDIS_TTL_DAYS", "7")),
        postgres_dsn=os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/customer_service"),
        postgres_retention_days=int(os.getenv("POSTGRES_RETENTION_DAYS", "90")),
        chroma_persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", ".chroma")),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "customer_service_kb"),
        chroma_shared_collection=os.getenv("CHROMA_SHARED_COLLECTION", "customer_service_shared_kb"),
        max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "12")),
        top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "4")),
        quality_threshold=int(os.getenv("QUALITY_THRESHOLD", "4")),
        rewrite_enabled=_to_bool(os.getenv("REWRITE_ENABLED"), default=True),
        workflow_name=os.getenv("WORKFLOW_NAME", "customer-service-workflow"),
        workflow_diagram_path=Path(os.getenv("WORKFLOW_DIAGRAM_PATH", "data/workflow/customer_service_workflow.mmd")),
    )