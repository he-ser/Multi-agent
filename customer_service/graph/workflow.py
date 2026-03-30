from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from customer_service.agents.experts import run_expert_agent
from customer_service.agents.retrieval_node import retrieve_context
from customer_service.agents.review import auto_approve, human_review, quality_check
from customer_service.agents.router import build_retrieval_filters, classify_intent, rewrite_query
from customer_service.config import get_settings
from customer_service.llm import build_chat_llm
from customer_service.observability import configure_langsmith
from customer_service.retrieval.hybrid import HybridRetriever
from customer_service.state import AgentState


def route_to_review(state: AgentState) -> str:
    return "human_review" if state.get("requires_human_review") else "auto_approve"


def _compile_graph(graph: StateGraph, workflow_name: str):
    try:
        return graph.compile(name=workflow_name)
    except TypeError:
        return graph.compile()


def build_customer_service_app(llm=None, retriever=None):
    settings = get_settings()
    configure_langsmith(settings)
    llm = llm or build_chat_llm()
    retriever = retriever or HybridRetriever()

    graph = StateGraph(AgentState)
    graph.add_node("rewrite_query", partial(rewrite_query, llm=llm))
    graph.add_node("classify_intent", partial(classify_intent, llm=llm))
    graph.add_node("build_retrieval_filters", build_retrieval_filters)
    graph.add_node("retrieve_context", partial(retrieve_context, retriever=retriever))
    graph.add_node("run_expert_agent", partial(run_expert_agent, llm=llm))
    graph.add_node("quality_check", partial(quality_check, llm=llm, threshold=settings.quality_threshold))
    graph.add_node("human_review", partial(human_review, llm=llm))
    graph.add_node("auto_approve", auto_approve)

    entry_node = "rewrite_query" if settings.rewrite_enabled else "classify_intent"
    graph.add_edge(START, entry_node)
    if settings.rewrite_enabled:
        graph.add_edge("rewrite_query", "classify_intent")
    graph.add_edge("classify_intent", "build_retrieval_filters")
    graph.add_edge("build_retrieval_filters", "retrieve_context")
    graph.add_edge("retrieve_context", "run_expert_agent")
    graph.add_edge("run_expert_agent", "quality_check")
    graph.add_conditional_edges(
        "quality_check",
        route_to_review,
        {
            "human_review": "human_review",
            "auto_approve": "auto_approve",
        },
    )
    graph.add_edge("human_review", END)
    graph.add_edge("auto_approve", END)
    return _compile_graph(graph, settings.workflow_name)
