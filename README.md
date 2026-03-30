# Multi-Agent Customer Service

基于 `LangGraph` 的多智能体客服系统，覆盖技术、销售、售后、反馈四类专家 Agent，并集成 `RAG`、`Chroma`、`Redis`、`PostgreSQL`、`LangSmith`。

## 特性

- 全部提示词已切换为中文，方便本地调试和后续业务扩展。
- 支持从 `PDF / DOCX / XLSX / CSV / TXT / MD` 自动解析、切片并入库。
- 支持 `share` 通用知识目录，所有智能体都会同时检索“专属知识 + 通用知识”。
- 专属知识与通用知识分别写入两个 Chroma collection，避免过滤逻辑互相污染。
- 支持多轮会话摘要与结构化事实记忆并存。
- Redis 会话消息和结构化记忆默认 7 天自动过期。
- PostgreSQL 会话记录支持按保留天数定期清理。
- LangSmith tracing 已接入，支持查看 LangGraph 节点执行链路、Prompt 和调用耗时。

## 知识库目录

结构化知识支持放在 `data/kb/*.jsonl`。

原始文档请放到 `data/kb/raw/<domain>/` 目录下：

```text
data/kb/raw/
  technical/
    api_manual.pdf
    faq.docx
  sales/
    pricing.xlsx
  support/
    refund_policy.docx
  feedback/
    summary.txt
  share/
    product_catalog.pdf
    company_policy.docx
```

## 环境变量

```env
ALIYUN_API_KEY=your-key
ALIYUN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ALIYUN_MODEL=deepseek-r1-0528
EMBEDDING_MODEL=text-embedding-v4
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=multi-agent-customer-service
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
WORKFLOW_NAME=customer-service-workflow
WORKFLOW_DIAGRAM_PATH=data/workflow/customer_service_workflow.mmd
REDIS_URL=redis://localhost:6379/0
REDIS_TTL_DAYS=7
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/customer_service
POSTGRES_RETENTION_DAYS=90
CHROMA_PERSIST_DIR=./.chroma
CHROMA_COLLECTION=customer_service_kb
CHROMA_SHARED_COLLECTION=customer_service_shared_kb
```

## 安装依赖

```powershell
python -m pip install -e .
```

## 启动基础服务

```powershell
docker compose up -d
```

## 构建 RAG 知识库

```powershell
python -m customer_service.scripts.ingest_kb --rebuild
```

## 运行项目

```powershell
python main.py
```

## 数据保留策略

- Redis 会话消息：默认 7 天自动过期
- Redis 结构化记忆：默认 7 天自动过期
- PostgreSQL 会话记录：默认保留 90 天

手动清理 PostgreSQL 过期记录：

```powershell
python -m customer_service.scripts.cleanup_postgres
```

如果你重新安装了项目脚本入口，也可以使用：

```powershell
customer-service-cleanup
```

## LangSmith 可视化

1. 在 LangSmith 控制台创建 API Key，并设置 `LANGSMITH_API_KEY`。
2. 设置 `LANGSMITH_TRACING=true`。
3. 运行 `main.py` 或调用 `CustomerService.chat(...)` 后，即可在 `LANGSMITH_PROJECT` 对应项目中查看链路。