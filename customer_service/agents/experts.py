from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from customer_service.prompts import EXPERT_PROMPTS
from customer_service.state import AgentState


def run_expert_agent(state: AgentState, llm) -> AgentState:
    intent = state.get("intent", "support")
    docs = state.get("retrieved_docs", [])
    context = "\n\n".join(
        [f"[{index + 1}] 内容：{doc['content']} | 元数据={doc['metadata']}" for index, doc in enumerate(docs)]
    ) or "未检索到匹配知识。请谨慎回答，必要时明确说明需要人工进一步处理。"

    structured_memory = {
        "user_facts": state.get("user_facts", {}),
        "ticket_context": state.get("ticket_context", {}),
    }

    prompt = [
        SystemMessage(content=EXPERT_PROMPTS.get(intent, EXPERT_PROMPTS["support"])),
        HumanMessage(
            content=(
                f"历史摘要：{state.get('memory_summary', '无')}\n\n"
                f"结构化记忆：{json.dumps(structured_memory, ensure_ascii=False)}\n\n"
                f"用户问题：{state['user_message']}\n\n"
                f"检索查询：{state.get('rewritten_query', state['user_message'])}\n\n"
                f"知识上下文：\n{context}\n\n"
                "请结合结构化记忆与知识上下文，生成一版专业、简洁、可执行的中文回复草稿。"
            )
        ),
    ]
    response = llm.invoke(prompt).content.strip()
    trace = list(state.get("trace", [])) + [f"expert={state.get('assigned_agent')}"]
    return {"draft_response": response, "trace": trace}