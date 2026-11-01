from __future__ import annotations

from modules.module3_adaptive_query_router.query_analyzer import QueryAnalyzer
from modules.module3_adaptive_query_router.routing_rules import RoutingRuleSet
from modules.module3_adaptive_query_router.schemas import RoutingResult


class AdaptiveQueryRouter:
    """Adaptive Query Router (AQR).

    This router does not call an LLM or perform retrieval. It simply analyzes the
    incoming query and emits a deterministic, explainable route. This is the first
    orchestration layer in CARO and is intentionally heuristic.
    """

    def __init__(self):
        self.analyzer = QueryAnalyzer()

    def analyze(self, query: str):
        return self.analyzer.analyze(query)

    def route(self, query: str) -> RoutingResult:
        analysis = self.analyzer.analyze(query)
        decision = RoutingRuleSet.decide(analysis.question_type, analysis)

        requires_retrieval = decision.route in {"RETRIEVE", "RETRIEVE_AND_MEMORY"}
        requires_memory = decision.route in {"MEMORY", "RETRIEVE_AND_MEMORY"}
        signals = {
            "procedural": analysis.has_procedural_signal,
            "comparison": analysis.has_comparison_signal,
            "multi_hop": analysis.has_multi_hop_signal,
            "contextual_reference": analysis.contains_contextual_marker,
            "knowledge_seeking": analysis.is_knowledge_seeking,
            "direct_answer": analysis.likely_direct_answer,
            "retrieval_required": requires_retrieval,
            "memory_required": requires_memory,
        }

        return RoutingResult(
            query=analysis.query,
            route=decision.route,
            question_type=analysis.question_type,
            requires_retrieval=requires_retrieval,
            requires_memory=requires_memory,
            confidence=round(float(decision.confidence), 2),
            reason=decision.reason,
            signals=signals,
        )

    def analyze_and_route(self, query: str) -> RoutingResult:
        return self.route(query)
