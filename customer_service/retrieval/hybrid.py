from __future__ import annotations

from collections import Counter
from typing import Iterable

from langchain_core.documents import Document

try:
    from langchain_chroma import Chroma
except Exception:  # pragma: no cover
    Chroma = None

from customer_service.config import get_settings
from customer_service.llm import build_embeddings
from customer_service.retrieval.knowledge_base import load_knowledge_records
from customer_service.retrieval.schemas import KnowledgeRecord
from customer_service.state import RetrievedDocument


def _tokenize(text: str) -> list[str]:
    normalized = text.replace(",", " ").replace(".", " ").replace(":", " ").replace(";", " ")
    return [token.lower() for token in normalized.split() if token.strip()]


class HybridRetriever:
    def __init__(
        self,
        vector_store: Chroma | None = None,
        shared_vector_store: Chroma | None = None,
        records: Iterable[KnowledgeRecord] | None = None,
    ) -> None:
        self.settings = get_settings()
        self.records = list(records) if records is not None else load_knowledge_records()
        self.vector_store = vector_store or self._build_store(self.settings.chroma_collection)
        self.shared_vector_store = shared_vector_store or self._build_store(self.settings.chroma_shared_collection)

    def _build_store(self, collection_name: str) -> Chroma | None:
        if Chroma is None:
            return None
        try:
            return Chroma(
                persist_directory=str(self.settings.chroma_persist_dir),
                collection_name=collection_name,
                embedding_function=build_embeddings(),
            )
        except Exception:
            return None

    def search(self, query: str, filters: dict[str, object] | None = None, top_k: int | None = None) -> list[RetrievedDocument]:
        filters = filters or {}
        top_k = top_k or self.settings.top_k_retrieval

        lexical_hits = self._keyword_search(query=query, filters=filters, top_k=top_k * 2)
        vector_hits = self._vector_search(self.vector_store, query=query, filters=filters, top_k=top_k * 2)
        shared_lexical_hits = self._keyword_search(query=query, filters={"domain": "share"}, top_k=top_k)
        shared_vector_hits = self._vector_search(self.shared_vector_store, query=query, filters={}, top_k=top_k)

        merged = self._merge_hits(lexical_hits, vector_hits, shared_lexical_hits, shared_vector_hits)
        return sorted(merged, key=lambda item: item["final_score"], reverse=True)[:top_k]

    def _vector_search(self, store: Chroma | None, query: str, filters: dict[str, object], top_k: int) -> list[RetrievedDocument]:
        if store is None:
            return []
        try:
            docs = store.similarity_search_with_relevance_scores(query, k=top_k, filter=filters or None)
        except Exception:
            return []

        results: list[RetrievedDocument] = []
        for doc, score in docs:
            results.append(
                {
                    "id": str(doc.metadata.get("id", "")),
                    "content": doc.page_content,
                    "metadata": dict(doc.metadata),
                    "vector_score": float(score),
                    "keyword_score": 0.0,
                    "final_score": float(score),
                }
            )
        return results

    def _keyword_search(self, query: str, filters: dict[str, object], top_k: int) -> list[RetrievedDocument]:
        query_tokens = Counter(_tokenize(query))
        results: list[RetrievedDocument] = []
        for record in self.records:
            if not self._matches_filters(record.metadata, filters):
                continue
            content_tokens = Counter(_tokenize(record.content))
            keyword_tokens = Counter(_tokenize(" ".join(record.metadata.get("keywords", []))))
            overlap = sum((query_tokens & (content_tokens + keyword_tokens)).values())
            if overlap <= 0:
                continue
            priority = float(record.metadata.get("priority", 1))
            keyword_score = overlap + priority * 0.1
            results.append(
                {
                    "id": record.id,
                    "content": record.content,
                    "metadata": dict(record.metadata),
                    "vector_score": 0.0,
                    "keyword_score": keyword_score,
                    "final_score": keyword_score,
                }
            )
        return sorted(results, key=lambda item: item["keyword_score"], reverse=True)[:top_k]

    @staticmethod
    def _matches_filters(metadata: dict[str, object], filters: dict[str, object]) -> bool:
        return all(metadata.get(key) == value for key, value in filters.items())

    @staticmethod
    def _merge_hits(*hit_groups: list[RetrievedDocument]) -> list[RetrievedDocument]:
        merged: dict[str, RetrievedDocument] = {}
        for group in hit_groups:
            for item in group:
                current = merged.get(item["id"])
                if current is None:
                    merged[item["id"]] = dict(item)
                    continue
                current["vector_score"] = max(current["vector_score"], item["vector_score"])
                current["keyword_score"] = max(current["keyword_score"], item["keyword_score"])
                current["final_score"] = current["vector_score"] * 0.65 + current["keyword_score"] * 0.35
        for item in merged.values():
            if item["final_score"] == 0.0:
                item["final_score"] = item["vector_score"] * 0.65 + item["keyword_score"] * 0.35
        return list(merged.values())