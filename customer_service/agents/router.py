from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from customer_service.prompts import REWRITE_PROMPT, ROUTER_PROMPT
from customer_service.state import AgentState


def rewrite_query(state: AgentState, llm) -> AgentState:
    user_message = state["user_message"]
    history = state.get("memory_summary", "")
    prompt = [
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=f"历史摘要：{history or '无'}\n\n最新用户问题：{user_message}"),
    ]
    rewritten_query = llm.invoke(prompt).content.strip()
    trace = list(state.get("trace", [])) + [f"rewritten_query={rewritten_query}"]
    return {"rewritten_query": rewritten_query, "trace": trace}


def classify_intent(state: AgentState, llm) -> AgentState:
    prompt = [
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=state.get("rewritten_query") or state["user_message"]),
    ]
    raw = llm.invoke(prompt).content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"intent": "unknown", "confidence": 0.0, "reason": f"invalid_router_output:{raw}"}

    intent = data.get("intent", "unknown")
    confidence = float(data.get("confidence", 0.0))
    reason = data.get("reason", "")
    route_map = {
        "technical": "technical_expert",
        "sales": "sales_expert",
        "support": "support_expert",
        "feedback": "feedback_expert",
        "unknown": "support_expert",
    }
    trace = list(state.get("trace", [])) + [f"intent={intent},confidence={confidence:.2f}"]
    return {
        "intent": intent,
        "intent_confidence": confidence,
        "routing_reason": reason,
        "assigned_agent": route_map.get(intent, "support_expert"),
        "trace": trace,
    }


def build_retrieval_filters(state: AgentState) -> AgentState:
    intent = state.get("intent", "unknown")
    filters = {}
    if intent != "unknown":
        filters["domain"] = intent
    trace = list(state.get("trace", [])) + [f"filters={filters}"]
    return {"retrieval_filters": filters, "trace": trace}