from __future__ import annotations

import uuid
from typing import Any

from customer_service.graph.workflow import build_customer_service_app
from customer_service.llm import build_chat_llm
from customer_service.memory.facts import extract_structured_memory
from customer_service.memory.redis_store import RedisConversationMemory
from customer_service.persistence.postgres import AsyncConversationRepository
from customer_service.retrieval.hybrid import HybridRetriever
from customer_service.state import AgentState, utc_now_iso


class CustomerService:
    def __init__(
        self,
        app=None,
        memory_store: RedisConversationMemory | None = None,
        repository: AsyncConversationRepository | None = None,
        retriever: HybridRetriever | None = None,
        memory_llm=None,
    ) -> None:
        self.memory_store = memory_store or RedisConversationMemory()
        self.repository = repository or AsyncConversationRepository()
        self.retriever = retriever or HybridRetriever()
        self.app = app or build_customer_service_app(retriever=self.retriever)
        self.memory_llm = memory_llm or build_chat_llm(temperature=0.0)

    def chat(self, user_message: str, session_id: str | None = None, user_id: str = "anonymous") -> dict[str, Any]:
        session_id = session_id or str(uuid.uuid4())
        history = self.memory_store.load_messages(session_id)
        memory_summary = self._summarize_history(history)
        structured_memory = self.memory_store.load_structured_memory(session_id)
        state: AgentState = {
            "session_id": session_id,
            "user_id": user_id,
            "user_message": user_message,
            "messages": history + [{"role": "user", "content": user_message, "timestamp": utc_now_iso()}],
            "memory_summary": memory_summary,
            "user_facts": dict(structured_memory.get("user_facts", {})),
            "ticket_context": dict(structured_memory.get("ticket_context", {})),
            "trace": [],
            "metadata": {"channel": "cli"},
        }
        result = self.app.invoke(state)
        user_turn = {"role": "user", "content": user_message, "timestamp": utc_now_iso()}
        assistant_turn = {
            "role": "assistant",
            "content": result["final_response"],
            "timestamp": utc_now_iso(),
        }
        self.memory_store.append_message(session_id, user_turn)
        self.memory_store.append_message(session_id, assistant_turn)

        # 这里把多轮对话中的稳定事实抽出来，避免后续只能依赖原始聊天记录。
        latest_structured_memory = self._update_structured_memory(
            existing_memory=structured_memory,
            user_message=user_message,
            assistant_message=result.get("final_response", ""),
        )
        self.memory_store.save_structured_memory(session_id, latest_structured_memory)
        result["user_facts"] = latest_structured_memory.get("user_facts", {})
        result["ticket_context"] = latest_structured_memory.get("ticket_context", {})

        self.repository.save_turn(
            {
                "session_id": session_id,
                "user_id": user_id,
                "intent": result.get("intent"),
                "assigned_agent": result.get("assigned_agent"),
                "requires_human_review": result.get("requires_human_review"),
                "rewritten_query": result.get("rewritten_query"),
                "retrieved_docs": result.get("retrieved_docs", []),
                "final_response": result.get("final_response"),
                "trace": result.get("trace", []),
                "user_facts": result.get("user_facts", {}),
                "ticket_context": result.get("ticket_context", {}),
            }
        )
        return result

    def _update_structured_memory(
        self,
        existing_memory: dict[str, Any],
        user_message: str,
        assistant_message: str,
    ) -> dict[str, Any]:
        try:
            return extract_structured_memory(
                llm=self.memory_llm,
                existing_memory=existing_memory,
                user_message=user_message,
                assistant_message=assistant_message,
            )
        except Exception:
            return existing_memory

    @staticmethod
    def _summarize_history(history: list[dict[str, Any]]) -> str:
        if not history:
            return ""
        clipped = history[-6:]
        return "\n".join([f"{item['role']}: {item['content']}" for item in clipped])