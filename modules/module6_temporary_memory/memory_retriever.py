from __future__ import annotations

import re

from .memory_store import LocalMemoryStore
from .schemas import FactRecord, MemoryResult, RetrievedMemory


class MemoryRetriever:
    """Semantic and factual relation retrieval over one temporary conversation scope."""

    def __init__(self, store: LocalMemoryStore, top_k: int = 5, threshold: float = 0.0):
        self.store = store
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, query: str, conversation_id: str, top_k: int | None = None, threshold: float | None = None) -> MemoryResult:
        effective_top_k = self.top_k if top_k is None else top_k
        effective_threshold = self.threshold if threshold is None else threshold
        matches = self.store.search(query, conversation_id, effective_top_k, effective_threshold)
        memories = [
            RetrievedMemory(
                memory_id=record.memory_id,
                conversation_id=record.conversation_id,
                turn_id=record.turn_id,
                similarity=similarity,
                original_query=record.user_query,
                stored_response=record.assistant_response,
                stored_context=record.context,
                timestamp=record.timestamp,
                metadata=record.metadata,
                retrieval_rank=rank,
                facts=getattr(record, "facts", []),
            )
            for rank, (similarity, record) in enumerate(matches, start=1)
        ]

        # Resolve specific structured facts relevant to the follow-up
        resolved_facts = self.store.search_facts(query, conversation_id)

        # Detect primary entity
        primary_entity = ""
        for mem in memories:
            text = f"{mem.original_query} {mem.stored_response}"
            ents = re.findall(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+\b", text)
            if ents:
                primary_entity = ents[0]
                break

        # Detect unresolved references
        unresolved = []
        for ref in ("those contributions", "which of those", "these contributions", "he", "his", "she", "her", "it", "they"):
            if ref in query.lower():
                unresolved.append(ref)

        return MemoryResult(
            query=query,
            conversation_id=conversation_id,
            memory_found=bool(memories or resolved_facts),
            memories=memories,
            memory_top_k=effective_top_k,
            memory_relevance_threshold=effective_threshold,
            message="Relevant temporary memory found." if (memories or resolved_facts) else "No relevant temporary memory found.",
            resolved_facts=resolved_facts,
            primary_entity=primary_entity,
            unresolved_references=unresolved,
        )
