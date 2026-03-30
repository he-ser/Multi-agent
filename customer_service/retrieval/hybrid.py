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
        docs, _ = self.search_with_debug(query=query, filters=filters, top_k=top_k)
        return docs

    def search_with_debug(
        self,
        query: str,
        filters: dict[str, object] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedDocument], list[str]]:
        filters = filters or {}
        top_k = top_k or self.settings.top_k_retrieval
        plans = self._build_filter_plans(filters)

        collected: list[RetrievedDocument] = []
        debug_steps: list[str] = []
        for plan_name, plan_filters in plans:
            lexical_hits = self._keyword_search(query=query, filters=plan_filters, top_k=top_k * 2)
            vector_hits = self._vector_search(self.vector_store, query=query, filters=plan_filters, top_k=top_k * 2)
            merged = self._merge_hits(lexical_hits, vector_hits)
            merged = sorted(merged, key=lambda item: item["final_score"], reverse=True)
            if merged:
                collected.extend(merged[:top_k])
                debug_steps.append(f"domain_plan={plan_name}:{len(merged[:top_k])}")
            if len(self._merge_hits(collected)) >= top_k:
                break
            debug_steps.append(f"domain_plan={plan_name}:0")

        shared_lexical_hits = self._keyword_search(query=query, filters={"domain": "share"}, top_k=top_k)
        shared_vector_hits = self._vector_search(self.shared_vector_store, query=query, filters={}, top_k=top_k)
        shared_hits = sorted(self._merge_hits(shared_lexical_hits, shared_vector_hits), key=lambda item: item["final_score"], reverse=True)
        if shared_hits:
            collected.extend(shared_hits[:top_k])
            debug_steps.append(f"shared_plan=share:{len(shared_hits[:top_k])}")
        else:
            debug_steps.append("shared_plan=share:0")

        final_hits = sorted(self._merge_hits(collected), key=lambda item: item["final_score"], reverse=True)[:top_k]
        return final_hits, debug_steps

    @staticmethod
    def _build_filter_plans(filters: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
        domain = filters.get("domain")
        product = filters.get("product")
        topic = filters.get("topic")
        plans: list[tuple[str, dict[str, object]]] = []

        if domain and product and topic:
            plans.append(("domain+product+topic", {"domain": domain, "product": product, "topic": topic}))
        if domain and product:
            plans.append(("domain+product", {"domain": domain, "product": product}))
        if domain:
            plans.append(("domain", {"domain": domain}))
        if not plans:
            plans.append(("no_filter", {}))

        unique_plans: list[tuple[str, dict[str, object]]] = []
        seen: set[tuple[tuple[str, object], ...]] = set()
        for name, plan_filters in plans:
            key = tuple(sorted(plan_filters.items()))
            if key in seen:
                continue
            seen.add(key)
            unique_plans.append((name, plan_filters))
        return unique_plans

    def _vector_search(self, store: Chroma | None, query: str, filters: dict[str, object], top_k: int) -> list[RetrievedDocument]:
        if store is None:
            return []
        try:
            docs = store.similarity_search_with_relevance_scores(query, k=top_k, filter=filters or None)
        except Exception:
            return []

        results: list[RetrievedDocument] = []
        for doc, score in docs:
            metadata = dict(doc.metadata)
            adjusted_score = float(score) + self._metadata_bonus(metadata, filters)
            results.append(
                {
                    "id": str(metadata.get("id", "")),
                    "content": doc.page_content,
                    "metadata": metadata,
                    "vector_score": adjusted_score,
                    "keyword_score": 0.0,
                    "final_score": adjusted_score,
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
            keyword_score = overlap + priority * 0.1 + self._metadata_bonus(record.metadata, filters)
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
    def _metadata_bonus(metadata: dict[str, object], filters: dict[str, object]) -> float:
        bonus = 0.0
        if filters.get("product") and metadata.get("product") == filters.get("product"):
            bonus += 0.35
        if filters.get("topic") and metadata.get("topic") == filters.get("topic"):
            bonus += 0.25
        return bonus

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