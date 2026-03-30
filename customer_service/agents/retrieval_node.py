from __future__ import annotations

from customer_service.retrieval.hybrid import HybridRetriever
from customer_service.state import AgentState


def retrieve_context(state: AgentState, retriever: HybridRetriever) -> AgentState:
    query = state.get("rewritten_query") or state["user_message"]
    filters = state.get("retrieval_filters", {})
    docs, debug_steps = retriever.search_with_debug(query=query, filters=filters)
    retrieval_notes = "已命中知识库" if docs else "未命中高置信度知识"
    trace = list(state.get("trace", [])) + [f"retrieved_docs={len(docs)}", *debug_steps]
    return {"retrieved_docs": docs, "retrieval_notes": retrieval_notes, "trace": trace}