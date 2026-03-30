from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from customer_service.prompts import MEMORY_EXTRACTION_PROMPT


def merge_memory(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "user_facts": dict(existing.get("user_facts", {})),
        "ticket_context": dict(existing.get("ticket_context", {})),
    }
    for section in ("user_facts", "ticket_context"):
        for key, value in dict(updates.get(section, {})).items():
            if value in (None, "", [], {}):
                continue
            merged[section][key] = value
    return merged


def extract_structured_memory(
    llm,
    existing_memory: dict[str, Any],
    user_message: str,
    assistant_message: str,
) -> dict[str, Any]:
    prompt = [
        SystemMessage(content=MEMORY_EXTRACTION_PROMPT),
        HumanMessage(
            content=(
                f"当前已知结构化记忆：{json.dumps(existing_memory, ensure_ascii=False)}\n\n"
                f"用户最新消息：{user_message}\n\n"
                f"助手最新回复：{assistant_message}"
            )
        ),
    ]
    raw = llm.invoke(prompt).content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return existing_memory

    return merge_memory(existing_memory, data)