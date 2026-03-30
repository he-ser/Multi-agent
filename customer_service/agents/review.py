from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from customer_service.prompts import HUMAN_REVIEW_PROMPT, QUALITY_PROMPT
from customer_service.state import AgentState


def quality_check(state: AgentState, llm, threshold: int) -> AgentState:
    prompt = [
        SystemMessage(content=QUALITY_PROMPT),
        HumanMessage(
            content=(
                f"用户问题：{state['user_message']}\n\n"
                f"回复草稿：{state.get('draft_response', '')}\n\n"
                f"命中知识条数：{len(state.get('retrieved_docs', []))}"
            )
        ),
    ]
    raw = llm.invoke(prompt).content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"score": 2, "reason": f"invalid_quality_output:{raw}", "requires_human_review": True}

    score = max(1, min(5, int(data.get("score", 2))))
    requires_human_review = bool(data.get("requires_human_review", score < threshold))
    if not state.get("retrieved_docs"):
        requires_human_review = True
    reason = str(data.get("reason", ""))
    escalation_reason = reason if requires_human_review else ""
    trace = list(state.get("trace", [])) + [f"quality={score},human={requires_human_review}"]
    return {
        "quality_score": score,
        "quality_reason": reason,
        "requires_human_review": requires_human_review,
        "escalation_reason": escalation_reason,
        "trace": trace,
    }


def human_review(state: AgentState, llm) -> AgentState:
    prompt = [
        SystemMessage(content=HUMAN_REVIEW_PROMPT),
        HumanMessage(
            content=(
                f"用户问题：{state['user_message']}\n\n"
                f"审核原因：{state.get('escalation_reason', '需要人工进一步核实')}\n\n"
                f"回复草稿：{state.get('draft_response', '')}"
            )
        ),
    ]
    reviewed = llm.invoke(prompt).content.strip()
    final = f"{reviewed}\n\n[已触发人工审核兜底]"
    return {"final_response": final}


def auto_approve(state: AgentState) -> AgentState:
    return {"final_response": state.get('draft_response', '')}