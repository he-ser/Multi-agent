from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.documents import Document

try:
    from langchain_chroma import Chroma
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 langchain-chroma，无法执行知识库入库。") from exc

from customer_service.config import get_settings
from customer_service.llm import build_embeddings
from customer_service.retrieval.knowledge_base import load_knowledge_records


def _delete_collection(persist_directory: str, collection_name: str, embeddings) -> None:
    try:
        existing = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        client = getattr(existing, "_client", None)
        if client is not None:
            client.delete_collection(collection_name)
    except Exception:
        pass


def _load_existing_ids(store: Chroma) -> set[str]:
    try:
        payload = store.get()
        return set(payload.get("ids", []))
    except Exception:
        return set()


def _add_records(store: Chroma, records, existing_ids: set[str]) -> int:
    docs = [
        Document(page_content=record.content, metadata={"id": record.id, **record.metadata})
        for record in records
        if record.id not in existing_ids
    ]
    if docs:
        store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])
    return len(docs)


def ingest_knowledge_base(
    kb_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    rebuild: bool = False,
) -> int:
    settings = get_settings()
    records = load_knowledge_records(kb_dir=kb_dir, raw_dir=raw_dir, include_raw=True)
    if not records:
        return 0

    persist_directory = str(settings.chroma_persist_dir)
    embeddings = build_embeddings()

    if rebuild:
        _delete_collection(persist_directory, settings.chroma_collection, embeddings)
        _delete_collection(persist_directory, settings.chroma_shared_collection, embeddings)

    domain_records = [record for record in records if record.metadata.get("domain") != "share"]
    shared_records = [record for record in records if record.metadata.get("domain") == "share"]

    domain_store = Chroma(
        persist_directory=persist_directory,
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
    )
    shared_store = Chroma(
        persist_directory=persist_directory,
        collection_name=settings.chroma_shared_collection,
        embedding_function=embeddings,
    )

    added_count = 0
    added_count += _add_records(domain_store, domain_records, _load_existing_ids(domain_store))
    added_count += _add_records(shared_store, shared_records, _load_existing_ids(shared_store))
    return added_count


def main() -> None:
    parser = argparse.ArgumentParser(description="将结构化知识和原始文档导入 Chroma。")
    parser.add_argument("--kb-dir", default="data/kb", help="结构化知识目录，默认 data/kb")
    parser.add_argument("--raw-dir", default="data/kb/raw", help="原始文档目录，默认 data/kb/raw")
    parser.add_argument("--rebuild", action="store_true", help="重建 collection 后重新入库")
    args = parser.parse_args()

    count = ingest_knowledge_base(kb_dir=args.kb_dir, raw_dir=args.raw_dir, rebuild=args.rebuild)
    print(f"知识库导入完成，新增 {count} 条记录。")


if __name__ == "__main__":
    main()