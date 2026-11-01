from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RoutingResult:
    """Structured output for a router decision."""

    query: str
    route: str
    question_type: str
    requires_retrieval: bool
    requires_memory: bool
    confidence: float
    reason: str
    signals: dict[str, bool] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
