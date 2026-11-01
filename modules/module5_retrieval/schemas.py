from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RetrievalConfig:
    """Development-time controls for retrieval and score-based filtering."""

    retrieval_top_k: int = 5
    final_top_n: int = 5
    relevance_threshold: float | None = None
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    max_documents: int | None = 5000
    dataset: str = "hotpotqa"
    index_type: str = "dev"
    index_path: str | None = None
    documents_path: str | None = None
    sparse_cache_path: str | None = None
    calibration_path: str | None = None
    calibration_query_limit: int = 128
    calibration_top_k: int = 10
    deduplicate_candidates: bool = True


@dataclass
class EvidenceDocument:
    """A candidate document with retrieval and filtering provenance."""

    document_id: str
    text: str
    retrieval_method: str
    retrieval_rank: int
    retrieval_score: float
    normalized_score: float
    selected_for_evidence: bool
    retrieved: bool = True
    filter_reason: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_score: float | None = None
    sparse_score: float | None = None
    normalized_dense_score: float | None = None
    normalized_sparse_score: float | None = None
    fusion_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """Structured Module 5 output passed to downstream modules."""

    query: str
    strategy: str
    candidate_count: int
    relevant_count: int
    unique_candidate_count: int = 0
    documents: list[EvidenceDocument] = field(default_factory=list)
    retrieval_executed: bool = True
    message: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    deduplicated_count: int = 0
    duplicate_document_ids: list[str] = field(default_factory=list)
    calibration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["documents"] = [document.to_dict() for document in self.documents]
        return payload
