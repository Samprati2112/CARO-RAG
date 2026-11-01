from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RetrievalStrategyResult:
    """Structured output for a Module 4 strategy decision."""

    query: str
    selected_strategy: str
    route: str
    question_type: str
    requires_retrieval: bool
    requires_memory: bool
    confidence: float
    reason: str
    signals: dict[str, bool]

    def to_dict(self):
        return asdict(self)