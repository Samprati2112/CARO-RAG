from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import RetrievalConfig


class DenseRetriever:
    """Adapter around Module 2's BGE-M3 and FAISS implementation."""

    def __init__(self, index_path: str | Path, config: RetrievalConfig | None = None, embedding_model=None, vector_store=None):
        self.index_path = Path(index_path)
        self.config = config or RetrievalConfig()
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if self.embedding_model is None:
            from modules.module2_baseline_rag.embeddings import EmbeddingModel
            self.embedding_model = EmbeddingModel()
        if self.vector_store is None:
            from modules.module2_baseline_rag.vector_store import FAISSVectorStore
            self.vector_store = FAISSVectorStore(self.index_path, top_k=top_k or self.config.retrieval_top_k)
        if self.vector_store.index is None:
            self.vector_store.load()
        query_vector = self.embedding_model.encode_query(query)
        results = self.vector_store.search(query_vector, top_k=top_k or self.config.retrieval_top_k)
        for rank, document in enumerate(results, start=1):
            document["retrieval_rank"] = rank
            document["dense_score"] = float(document["retrieval_score"])
            document["normalized_score"] = float(document["retrieval_score"])
        return results
