from __future__ import annotations

import json

from customer_service.graph.workflow import build_customer_service_app
from customer_service.retrieval.hybrid import HybridRetriever


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def invoke(self, messages):
        text = "\n".join([getattr(message, "content", str(message)) for message in messages])
        if "检索查询改写器" in text:
            return FakeResponse("API key 401 error after configuration")
        if "多智能体客服路由器" in text:
            return FakeResponse(json.dumps({"intent": "technical", "confidence": 0.93, "reason": "这是一个 API 鉴权报错问题"}, ensure_ascii=False))
        if "回复质量审核助手" in text:
            return FakeResponse(json.dumps({"score": 5, "reason": "回答完整准确", "requires_human_review": False}, ensure_ascii=False))
        if "人工审核兜底助手" in text:
            return FakeResponse("这是人工审核后的回复")
        return FakeResponse("请检查 API Key、Authorization 请求头、Base URL 以及接口权限配置。")


def test_workflow_routes_to_technical_and_returns_response():
    app = build_customer_service_app(llm=FakeLLM(), retriever=HybridRetriever())
    result = app.invoke(
        {
            "session_id": "s1",
            "user_id": "u1",
            "user_message": "API 一直返回 401，我应该检查什么？",
            "messages": [{"role": "user", "content": "API 一直返回 401，我应该检查什么？"}],
            "memory_summary": "",
            "trace": [],
            "metadata": {},
        }
    )

    assert result["intent"] == "technical"
    assert result["assigned_agent"] == "technical_expert"
    assert "API Key" in result["final_response"]
    assert result["requires_human_review"] is False