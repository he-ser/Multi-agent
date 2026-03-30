from __future__ import annotations

import os

from customer_service.config import Settings


def configure_langsmith(settings: Settings) -> None:
    if not settings.langsmith_tracing:
        return

    env_updates = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_PROJECT": settings.langsmith_project,
        "LANGSMITH_ENDPOINT": settings.langsmith_endpoint,
        # Older LangChain integrations still read these names.
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_PROJECT": settings.langsmith_project,
        "LANGCHAIN_ENDPOINT": settings.langsmith_endpoint,
    }
    if settings.langsmith_api_key:
        env_updates["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        env_updates["LANGCHAIN_API_KEY"] = settings.langsmith_api_key

    for key, value in env_updates.items():
        os.environ[key] = value
