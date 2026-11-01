from __future__ import annotations

from pathlib import Path
from typing import Any

from config.config import MEMORY_DIR

from .memory_retriever import MemoryRetriever
from .memory_store import LocalMemoryStore
from .schemas import MemoryRecord, MemoryResult


class TemporaryMemoryManager:
    """Public Module 6 API for temporary task/conversation memory."""

    def __init__(self, storage_path: str | Path = MEMORY_DIR, embedder=None, memory_top_k: int = 5, memory_relevance_threshold: float = 0.0):
        self.store = LocalMemoryStore(storage_path, embedder=embedder)
        self.retriever = MemoryRetriever(self.store, memory_top_k, memory_relevance_threshold)

    def add_memory(
        self,
        conversation_id: str,
        turn_id: str,
        user_query: str,
        assistant_response: str = "",
        context: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        return self.store.add(conversation_id, turn_id, user_query, assistant_response, context, metadata)

    def retrieve_memory(
        self,
        query: str,
        conversation_id: str,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        strategy: str | None = None,
    ) -> MemoryResult:
        if strategy not in {None, "MEMORY", "MEMORY_AND_RETRIEVAL"}:
            raise ValueError("Module 6 memory retrieval accepts MEMORY or MEMORY_AND_RETRIEVAL strategies")
        return self.retriever.retrieve(query, conversation_id, top_k, threshold)

    def clear_memory(self, conversation_id: str | None = None) -> int:
        return self.store.clear(conversation_id)

    def get_memory_state(self, conversation_id: str | None = None) -> dict[str, Any]:
        return self.store.state(conversation_id)
