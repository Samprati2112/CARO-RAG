from __future__ import annotations

from modules.module3_adaptive_query_router.schemas import RoutingResult

from .schemas import RetrievalStrategyResult
from .strategy_rules import RetrievalStrategyRules


class RetrievalStrategySelector:
    """Select a retrieval strategy from an existing Module 3 routing result."""

    def select(self, routing_result: RoutingResult) -> RetrievalStrategyResult:
        if not isinstance(routing_result, RoutingResult):
            raise TypeError("routing_result must be a Module 3 RoutingResult")
        return RetrievalStrategyRules.select(routing_result)