from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from customer_service.prompts import MEMORY_EXTRACTION_PROMPT
from customer_service.state import utc_now_iso

USER_FACT_FIELDS = {
    "company_name",
    "contact_name",
    "contact_email",
    "team_size",
    "product",
    "deployment",
}

TICKET_CONTEXT_FIELDS = {
    "issue_type",
    "error_code",
    "product",
    "topic",
    "order_id",
    "latest_request",
}

PROTECTED_USER_FIELDS = {"company_name", "contact_email", "team_size"}
PROTECTED_TICKET_FIELDS = {"error_code", "order_id"}
ALLOWED_PRODUCTS = {"api", "enterprise", "billing", "account", "platform"}
ALLOWED_TOPICS = {"auth", "pricing", "refund", "security", "roadmap", "general"}


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_email(value: str) -> str:
    return value.lower() if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.lower()) else ""


def _normalize_product(value: str) -> str:
    value = value.lower()
    return value if value in ALLOWED_PRODUCTS else ""


def _normalize_topic(value: str) -> str:
    value = value.lower()
    return value if value in ALLOWED_TOPICS else ""


def _normalize_team_size(value: str) -> str:
    return value if value else ""


def _normalize_error_code(value: str) -> str:
    value = value.upper()
    match = re.search(r"[A-Z]*\d{3,5}", value)
    return match.group(0) if match else value[:32]


def _normalize_order_id(value: str) -> str:
    return value[:64]


def _normalize_memory_section(section: dict[str, Any], allowed_fields: set[str], section_name: str) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, raw_value in dict(section).items():
        if key not in allowed_fields:
            continue
        value = _clean_value(raw_value)
        if not value:
            continue
        if key == "contact_email":
            value = _normalize_email(value)
        elif key == "product":
            value = _normalize_product(value)
        elif key == "topic":
            value = _normalize_topic(value)
        elif key == "team_size":
            value = _normalize_team_size(value)
        elif key == "error_code":
            value = _normalize_error_code(value)
        elif key == "order_id":
            value = _normalize_order_id(value)
        if not value:
            continue
        normalized[key] = value

    if section_name == "ticket_context" and "latest_request" in normalized:
        normalized["latest_request"] = normalized["latest_request"][:120]
    return normalized


def _should_overwrite(section: str, key: str, old_value: str, new_value: str) -> bool:
    if not old_value:
        return True
    if old_value == new_value:
        return False
    protected_fields = PROTECTED_USER_FIELDS if section == "user_facts" else PROTECTED_TICKET_FIELDS
    if key in protected_fields and len(new_value) < len(old_value):
        return False
    return True


def _empty_memory() -> dict[str, Any]:
    return {"user_facts": {}, "ticket_context": {}, "user_facts_meta": {}, "ticket_context_meta": {}}


def merge_memory(
    existing: dict[str, Any],
    updates: dict[str, Any],
    *,
    source_turn: int | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    merged = _empty_memory()
    merged["user_facts"].update(dict(existing.get("user_facts", {})))
    merged["ticket_context"].update(dict(existing.get("ticket_context", {})))
    merged["user_facts_meta"].update(dict(existing.get("user_facts_meta", {})))
    merged["ticket_context_meta"].update(dict(existing.get("ticket_context_meta", {})))

    normalized_updates = {
        "user_facts": _normalize_memory_section(updates.get("user_facts", {}), USER_FACT_FIELDS, "user_facts"),
        "ticket_context": _normalize_memory_section(updates.get("ticket_context", {}), TICKET_CONTEXT_FIELDS, "ticket_context"),
    }
    update_ts = updated_at or utc_now_iso()

    for section in ("user_facts", "ticket_context"):
        meta_section = "user_facts_meta" if section == "user_facts" else "ticket_context_meta"
        for key, value in normalized_updates[section].items():
            old_value = _clean_value(merged[section].get(key, ""))
            if _should_overwrite(section, key, old_value, value):
                merged[section][key] = value
                merged[meta_section][key] = {
                    "updated_at": update_ts,
                    "source_turn": source_turn,
                }
    return merged


def extract_structured_memory(
    llm,
    existing_memory: dict[str, Any],
    user_message: str,
    assistant_message: str,
    *,
    source_turn: int | None = None,
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

    return merge_memory(existing_memory, data, source_turn=source_turn)