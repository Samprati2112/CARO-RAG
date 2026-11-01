from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """Deterministic validation status for a generated answer and its citations."""

    is_valid: bool
    is_grounded: bool
    grounding_score: float
    status: str  # "VALIDATED", "UNGROUNDED", "ABSTAINED", "MALFORMED", "EMPTY", "INVALID_CITATION"
    reasons: list[str] = field(default_factory=list)
    unsupported_entities: list[str] = field(default_factory=list)
    claim_validations: list[dict[str, Any]] = field(default_factory=list)
    citation_accuracy_score: float = 1.0
    correction_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationMetrics:
    """Standardized generation and grounding evaluation metrics."""

    exact_match: float = 0.0
    f1_score: float = 0.0
    grounding_ratio: float = 0.0
    context_relevance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerGenerationResult:
    """Complete Module 8 structured result."""

    query: str
    raw_response: str
    final_answer: str
    is_grounded: bool
    grounding_score: float
    validation: ValidationResult
    evaluation: EvaluationMetrics | None = None
    evidence_count: int = 0
    model_name: str = "gemma3:4b"
    citation_status: str = "VALIDATED"  # "VALIDATED", "CORRECTED", "PARTIAL", "REJECTED", "ABSTAINED"
    provenance_summary: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "raw_response": self.raw_response,
            "final_answer": self.final_answer,
            "is_grounded": self.is_grounded,
            "grounding_score": self.grounding_score,
            "validation": self.validation.to_dict(),
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "evidence_count": self.evidence_count,
            "model_name": self.model_name,
            "citation_status": self.citation_status,
            "provenance_summary": self.provenance_summary,
        }
