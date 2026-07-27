"""
Hybrid retrieval module for MediGuide-RAG.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """Combines dense retrieval and BM25 with RRF reranking."""

    def __init__(
        self,
        vectorstore: FAISS,
        chunks: List[Document],
        use_cross_encoder_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-base",
    ):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.use_cross_encoder_reranker = use_cross_encoder_reranker
        self.reranker_model = reranker_model
        self.cross_encoder = None
        self.setup_retrievers()
        self.setup_reranker()

    def setup_retrievers(self):
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8},
        )
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks, k=8)

    def setup_reranker(self):
        if not self.use_cross_encoder_reranker:
            return
        try:
            from sentence_transformers import CrossEncoder

            self.cross_encoder = CrossEncoder(self.reranker_model)
            logger.info("Loaded Cross-Encoder reranker: %s", self.reranker_model)
        except Exception as exc:  # pragma: no cover - optional runtime dependency/model
            logger.warning("Cross-Encoder reranker disabled: %s", exc)
            self.cross_encoder = None

    def hybrid_search(self, query: str, top_k: int = 4) -> List[Document]:
        candidate_k = max(8, top_k * 2)
        vector_docs, bm25_docs = self._retrieve_candidates(query, candidate_k)
        candidates = self._rrf_rerank(vector_docs, bm25_docs, query=query)
        candidates = self._cross_encoder_rerank(query, candidates)
        return self._dedupe_by_parent(candidates)[:top_k]

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 4) -> List[Document]:
        candidate_k = max(64, top_k * 16)
        vector_docs, bm25_docs = self._retrieve_candidates(query, candidate_k)
        candidates = self._rrf_rerank(vector_docs, bm25_docs, query=query)
        candidates = self._cross_encoder_rerank(query, candidates)
        filtered = []
        for doc in candidates:
            matched = True
            for key, value in filters.items():
                doc_value = doc.metadata.get(key)
                if isinstance(value, list):
                    if doc_value not in value:
                        matched = False
                        break
                elif doc_value != value:
                    matched = False
                    break
            if matched:
                filtered.append(doc)
                if len(filtered) >= top_k:
                    break
        if filtered:
            return filtered
        logger.info("Metadata filters returned no results; falling back to unfiltered hybrid retrieval")
        return self._dedupe_by_parent(candidates)[:top_k]

    def _retrieve_candidates(self, query: str, candidate_k: int):
        vector_docs = self.vectorstore.similarity_search(query, k=candidate_k)
        previous_k = self.bm25_retriever.k
        self.bm25_retriever.k = candidate_k
        try:
            bm25_docs = self.bm25_retriever.invoke(query)
        finally:
            self.bm25_retriever.k = previous_k
        return vector_docs, bm25_docs

    def _rrf_rerank(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = 60,
        query: str = "",
    ) -> List[Document]:
        scores = {}
        objects = {}

        for source, docs in (("vector", vector_docs), ("bm25", bm25_docs)):
            for rank, doc in enumerate(docs, start=1):
                doc_id = doc.metadata.get("chunk_id") or hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
                objects[doc_id] = doc
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
                doc.metadata.setdefault("retrieval_sources", [])
                doc.metadata["retrieval_sources"].append(source)

        lowered_query = query.casefold()
        for doc_id, doc in objects.items():
            focus = str(doc.metadata.get("focus", "")).strip().casefold()
            if len(focus) >= 4 and focus in lowered_query:
                scores[doc_id] += 1.0 / (k + 1)
                doc.metadata["exact_focus_match"] = True

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results = []
        for doc_id, score in ranked:
            doc = objects[doc_id]
            doc.metadata["rrf_score"] = score
            results.append(doc)
        logger.info("RRF rerank: vector=%s bm25=%s merged=%s", len(vector_docs), len(bm25_docs), len(results))
        return results

    def _cross_encoder_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not self.cross_encoder or not docs:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        for doc, score in zip(docs, scores):
            doc.metadata["cross_encoder_score"] = float(score)
        return sorted(
            docs,
            key=lambda doc: (
                float(doc.metadata.get("cross_encoder_score", 0.0)),
                float(doc.metadata.get("rrf_score", 0.0)),
            ),
            reverse=True,
        )

    @staticmethod
    def _dedupe_by_parent(docs: List[Document]) -> List[Document]:
        seen = set()
        unique = []
        for doc in docs:
            parent_id = doc.metadata.get("parent_id") or doc.metadata.get("chunk_id")
            if parent_id in seen:
                continue
            seen.add(parent_id)
            unique.append(doc)
        return unique
