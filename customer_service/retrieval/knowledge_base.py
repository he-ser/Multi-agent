from __future__ import annotations

import json
from pathlib import Path

from customer_service.retrieval.document_loader import load_raw_documents
from customer_service.retrieval.schemas import KnowledgeRecord


DEFAULT_KNOWLEDGE_BASE: list[KnowledgeRecord] = [
    KnowledgeRecord(
        id="tech_api_401",
        content="当接口返回 401 时，通常表示 API Key 无效、Authorization 请求头缺失、Base URL 配置错误，或者当前账号没有目标接口的访问权限。排查时应同时核对密钥、请求头、调用地址和模型授权范围。",
        metadata={"domain": "technical", "agent": "technical_expert", "product": "api", "topic": "auth", "priority": 10, "keywords": ["401", "api key", "authorization", "鉴权", "权限"]},
    ),
    KnowledgeRecord(
        id="sales_enterprise_plan",
        content="企业版通常面向更大规模团队，常见能力包括更高并发、专属成功经理、审计日志和 SLA 支持。最终报价应由销售根据座席规模、调用量和交付要求评估后提供。",
        metadata={"domain": "sales", "agent": "sales_expert", "product": "enterprise", "topic": "pricing", "priority": 8, "keywords": ["企业版", "报价", "pricing", "套餐", "plan"]},
    ),
    KnowledgeRecord(
        id="support_refund_sla",
        content="退款申请通常会在 1 到 3 个工作日内完成审核。如果涉及发票作废、对公付款或特殊支付渠道，处理时间可能更长。客服应先收集订单号、申请时间和支付方式。",
        metadata={"domain": "support", "agent": "support_expert", "product": "billing", "topic": "refund", "priority": 9, "keywords": ["退款", "订单号", "refund", "发票"]},
    ),
    KnowledgeRecord(
        id="feedback_loop",
        content="产品反馈会进入统一评审队列，产品团队通常会根据业务价值、实现成本和客户影响范围进行优先级排序。高频且影响面广的需求更容易进入路线图讨论。",
        metadata={"domain": "feedback", "agent": "feedback_expert", "product": "platform", "topic": "roadmap", "priority": 6, "keywords": ["反馈", "需求", "路线图", "roadmap"]},
    ),
]


def _default_kb_dir() -> Path:
    return Path("data/kb")


def _default_raw_dir() -> Path:
    return Path("data/kb/raw")


def load_knowledge_records(
    kb_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    include_raw: bool = True,
) -> list[KnowledgeRecord]:
    base_dir = Path(kb_dir) if kb_dir else _default_kb_dir()
    records: list[KnowledgeRecord] = []

    if base_dir.exists():
        for file_path in sorted(base_dir.glob("*.jsonl")):
            for line in file_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                metadata = dict(payload.get("metadata", {}))
                metadata.setdefault("source", file_path.name)
                records.append(
                    KnowledgeRecord(
                        id=str(payload["id"]),
                        content=str(payload["content"]),
                        metadata=metadata,
                    )
                )

    if include_raw:
        raw_records = load_raw_documents(raw_dir or _default_raw_dir())
        existing_ids = {record.id for record in records}
        records.extend([record for record in raw_records if record.id not in existing_ids])

    return records or list(DEFAULT_KNOWLEDGE_BASE)