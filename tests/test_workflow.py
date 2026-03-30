from __future__ import annotations

import json

from customer_service.graph.workflow import build_customer_service_app
from customer_service.memory.facts import merge_memory
from customer_service.retrieval.hybrid import HybridRetriever
from customer_service.retrieval.schemas import KnowledgeRecord
from customer_service.services.chat import CustomerService


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def invoke(self, messages):
        text = "\n".join([getattr(message, "content", str(message)) for message in messages])
        if "结构化记忆抽取器" in text:
            return FakeResponse(
                json.dumps(
                    {
                        "user_facts": {
                            "company_name": "星河科技",
                            "contact_name": "",
                            "contact_email": "ops@xinghe.ai",
                            "team_size": "50人",
                            "product": "api",
                            "deployment": "公有云"
                        },
                        "ticket_context": {
                            "issue_type": "鉴权失败",
                            "error_code": "401",
                            "product": "api",
                            "topic": "auth",
                            "order_id": "ORD-1001",
                            "latest_request": "排查 API 401 问题"
                        }
                    },
                    ensure_ascii=False,
                )
            )
        if "检索查询改写器" in text:
            return FakeResponse("API key 401 error after configuration")
        if "多智能体客服路由器" in text:
            return FakeResponse(
                json.dumps(
                    {
                        "intent": "technical",
                        "confidence": 0.93,
                        "reason": "这是一个 API 鉴权报错问题",
                        "product": "api",
                        "topic": "auth"
                    },
                    ensure_ascii=False,
                )
            )
        if "回复质量审核助手" in text:
            return FakeResponse(json.dumps({"score": 5, "reason": "回答完整准确", "requires_human_review": False}, ensure_ascii=False))
        if "人工审核兜底助手" in text:
            return FakeResponse("这是人工审核后的回复")
        return FakeResponse("请检查 API Key、Authorization 请求头、Base URL 以及接口权限配置。")


class InMemoryRepository:
    def __init__(self) -> None:
        self.items = []

    def save_turn(self, payload):
        self.items.append(payload)


class InMemoryConversationMemory:
    def __init__(self) -> None:
        self.messages = {}
        self.structured = {}

    def load_messages(self, session_id: str):
        return list(self.messages.get(session_id, []))

    def append_message(self, session_id: str, message):
        self.messages.setdefault(session_id, []).append(message)

    def load_structured_memory(self, session_id: str):
        return dict(self.structured.get(session_id, {"user_facts": {}, "ticket_context": {}, "user_facts_meta": {}, "ticket_context_meta": {}}))

    def save_structured_memory(self, session_id: str, structured_memory):
        self.structured[session_id] = structured_memory


def test_workflow_routes_to_technical_and_returns_response():
    app = build_customer_service_app(llm=FakeLLM(), retriever=HybridRetriever(vector_store=None, shared_vector_store=None))
    result = app.invoke(
        {
            "session_id": "s1",
            "user_id": "u1",
            "user_message": "API 一直返回 401，我应该检查什么？",
            "messages": [{"role": "user", "content": "API 一直返回 401，我应该检查什么？"}],
            "memory_summary": "",
            "user_facts": {},
            "ticket_context": {},
            "user_facts_meta": {},
            "ticket_context_meta": {},
            "trace": [],
            "metadata": {},
        }
    )

    assert result["intent"] == "technical"
    assert result["assigned_agent"] == "technical_expert"
    assert result["product"] == "api"
    assert result["topic"] == "auth"
    assert result["retrieval_filters"] == {"domain": "technical", "product": "api", "topic": "auth"}
    assert any(step.startswith("domain_plan=") for step in result["trace"])
    assert "API Key" in result["final_response"]
    assert result["requires_human_review"] is False


def test_retriever_falls_back_when_topic_filter_is_too_strict():
    retriever = HybridRetriever(
        vector_store=None,
        shared_vector_store=None,
        records=[
            KnowledgeRecord(
                id="tech_api_general",
                content="API 鉴权失败时应先检查 API Key、Authorization 请求头和接口权限。",
                metadata={"domain": "technical", "product": "api", "topic": "general", "priority": 10, "keywords": ["api", "鉴权", "401"]},
            ),
            KnowledgeRecord(
                id="share_policy",
                content="所有客服回复都应保持准确、克制并说明边界。",
                metadata={"domain": "share", "agent": "shared", "priority": 5, "keywords": ["客服", "回复"]},
            ),
        ],
    )

    docs, debug_steps = retriever.search_with_debug(
        query="API 返回 401，鉴权失败怎么办？",
        filters={"domain": "technical", "product": "api", "topic": "auth"},
        top_k=3,
    )

    assert docs
    assert docs[0]["id"] == "tech_api_general"
    assert "domain_plan=domain+product+topic:0" in debug_steps
    assert any(step.startswith("domain_plan=domain+product:") for step in debug_steps)


def test_customer_service_updates_structured_memory_and_repository_fields():
    memory_store = InMemoryConversationMemory()
    repository = InMemoryRepository()
    llm = FakeLLM()
    service = CustomerService(
        app=build_customer_service_app(llm=llm, retriever=HybridRetriever(vector_store=None, shared_vector_store=None)),
        memory_store=memory_store,
        repository=repository,
        retriever=HybridRetriever(vector_store=None, shared_vector_store=None),
        memory_llm=llm,
    )

    result = service.chat(user_message="我们是星河科技，50人团队，API 返回 401。", session_id="demo", user_id="u1")

    assert result["user_facts"]["company_name"] == "星河科技"
    assert result["user_facts_meta"]["company_name"]["source_turn"] == 1
    assert result["ticket_context"]["error_code"] == "401"
    assert result["ticket_context_meta"]["error_code"]["source_turn"] == 1
    assert memory_store.load_structured_memory("demo")["ticket_context_meta"]["topic"]["source_turn"] == 1
    assert repository.items
    saved = repository.items[0]
    assert saved["product"] == "api"
    assert saved["topic"] == "auth"
    assert saved["error_code"] == "401"
    assert saved["order_id"] == "ORD-1001"
    assert saved["company_name"] == "星河科技"
    assert saved["contact_email"] == "ops@xinghe.ai"
    assert saved["team_size"] == "50人"
    assert saved["user_facts_meta"]["company_name"]["source_turn"] == 1


def test_merge_memory_applies_validation_and_update_rules():
    existing = {
        "user_facts": {
            "company_name": "星河科技有限公司",
            "contact_email": "ops@xinghe.ai",
            "team_size": "50人",
            "product": "api",
        },
        "ticket_context": {
            "error_code": "E1001",
            "order_id": "ORD-9999",
            "topic": "auth",
        },
        "user_facts_meta": {"company_name": {"updated_at": "2026-03-30T00:00:00+00:00", "source_turn": 1}},
        "ticket_context_meta": {"order_id": {"updated_at": "2026-03-30T00:00:00+00:00", "source_turn": 1}},
    }
    updates = {
        "user_facts": {
            "company_name": "星河",
            "contact_email": "invalid-email",
            "team_size": "80人",
            "product": "unknown-product",
            "contact_name": "张三",
            "unexpected": "should_drop",
        },
        "ticket_context": {
            "error_code": "401",
            "order_id": "1",
            "topic": "invalid-topic",
            "latest_request": "请尽快帮我排查 API 鉴权失败问题，影响今天上线。",
        },
    }

    merged = merge_memory(existing, updates, source_turn=2, updated_at="2026-03-31T00:00:00+00:00")

    assert merged["user_facts"]["company_name"] == "星河科技有限公司"
    assert merged["user_facts"]["contact_email"] == "ops@xinghe.ai"
    assert merged["user_facts"]["team_size"] == "80人"
    assert merged["user_facts"]["product"] == "api"
    assert merged["user_facts"]["contact_name"] == "张三"
    assert "unexpected" not in merged["user_facts"]
    assert merged["ticket_context"]["error_code"] == "401"
    assert merged["ticket_context"]["order_id"] == "ORD-9999"
    assert merged["ticket_context"]["topic"] == "auth"
    assert merged["ticket_context"]["latest_request"].startswith("请尽快帮我排查")
    assert merged["user_facts_meta"]["team_size"]["source_turn"] == 2
    assert merged["ticket_context_meta"]["error_code"]["source_turn"] == 2
    assert merged["ticket_context_meta"]["order_id"]["source_turn"] == 1