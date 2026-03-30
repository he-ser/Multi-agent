from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


Intent = Literal["technical", "sales", "support", "feedback", "unknown"]
ExpertAgentName = Literal["technical_expert", "sales_expert", "support_expert", "feedback_expert", "unknown"]


class RetrievedDocument(TypedDict):
    id: str
    content: str
    metadata: dict[str, Any]
    vector_score: float
    keyword_score: float
    final_score: float


class AgentState(TypedDict, total=False):
    session_id: str
    user_id: str
    user_message: str
    rewritten_query: str
    intent: Intent
    intent_confidence: float
    assigned_agent: ExpertAgentName
    routing_reason: str
    product: str
    topic: str
    messages: Annotated[list[dict[str, Any]], add_messages]
    memory_summary: str
    retrieved_docs: list[RetrievedDocument]
    retrieval_filters: dict[str, Any]
    retrieval_notes: str
    draft_response: str
    quality_score: int
    quality_reason: str
    requires_human_review: bool
    escalation_reason: str
    final_response: str
    trace: list[str]
    metadata: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()