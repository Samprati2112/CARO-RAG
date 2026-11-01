from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FactRecord:
    """An individual structured factual claim stored in conversation memory."""

    fact_id: str
    conversation_id: str
    turn_id: str
    entity: str
    relation: str
    value: str
    provenance: str = "CORPUS_VERIFIED"  # "CORPUS_VERIFIED", "PARAMETRIC", "INFERRED", "UNVERIFIED"
    source_turn: str = ""
    source_text: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryRecord:
    """A temporary, conversation-scoped turn stored by Module 6."""

    memory_id: str
    conversation_id: str
    turn_id: str
    user_query: str
    assistant_response: str
    context: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list, repr=False)
    facts: list[FactRecord] = field(default_factory=list)

    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_embedding:
            payload.pop("embedding", None)
        payload["facts"] = [f.to_dict() if hasattr(f, "to_dict") else f for f in self.facts]
        return payload


@dataclass
class RetrievedMemory:
    memory_id: str
    conversation_id: str
    turn_id: str
    similarity: float
    original_query: str
    stored_response: str
    stored_context: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_rank: int = 0
    facts: list[FactRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["facts"] = [f.to_dict() if hasattr(f, "to_dict") else f for f in self.facts]
        return payload


@dataclass
class MemoryResult:
    query: str
    conversation_id: str
    memory_found: bool
    memories: list[RetrievedMemory] = field(default_factory=list)
    memory_top_k: int = 5
    memory_relevance_threshold: float = 0.0
    message: str = ""
    resolved_facts: list[FactRecord] = field(default_factory=list)
    primary_entity: str = ""
    unresolved_references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["memories"] = [memory.to_dict() for memory in self.memories]
        payload["resolved_facts"] = [f.to_dict() if hasattr(f, "to_dict") else f for f in self.resolved_facts]
        return payload
