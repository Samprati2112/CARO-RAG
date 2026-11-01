from __future__ import annotations

import re

from modules.module3_adaptive_query_router.schemas import RoutingResult

from .schemas import RetrievalStrategyResult


class RetrievalStrategyRules:
    """Deterministic rules that map Module 3 routing to a retrieval strategy."""

    @staticmethod
    def _signals(routing_result: RoutingResult) -> dict[str, bool]:
        query = routing_result.query
        lower = query.lower()
        has_multi_hop = routing_result.question_type == "multi-hop"
        has_keyword_sensitive_terms = bool(
            re.search(
                r"\b(population|isbn|postcode|zip code|release date|official name|born|birthplace|located|founded)\b",
                lower,
            )
        )
        has_named_entity = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query))
        return {
            "multi_hop": has_multi_hop,
            "keyword_sensitive_terms": has_keyword_sensitive_terms,
            "named_entity": has_named_entity,
        }

    @classmethod
    def select(cls, routing_result: RoutingResult) -> RetrievalStrategyResult:
        signals = cls._signals(routing_result)
        requires_retrieval = routing_result.requires_retrieval
        requires_memory = routing_result.requires_memory

        if not requires_retrieval and not requires_memory:
            strategy = "DIRECT"
            confidence = 0.95
            reason = (
                f"M3 identified a {routing_result.question_type} query with no retrieval requirement and no conversational-memory requirement "
                "under the current routing rules. Therefore, no retrieval strategy is needed and DIRECT is selected."
            )
        elif requires_retrieval and requires_memory:
            strategy = "MEMORY_AND_RETRIEVAL"
            confidence = 0.92
            reason = (
                f"M3 identified a {routing_result.question_type} query that requires both previous conversational context and external knowledge. "
                "Therefore, MEMORY_AND_RETRIEVAL is selected so prior context can be incorporated while external evidence is retrieved."
            )
        elif requires_memory:
            strategy = "MEMORY"
            confidence = 0.88
            reason = (
                f"M3 identified a {routing_result.question_type} query that depends on previous conversational context and does not currently "
                "indicate a need for external retrieval. Therefore, MEMORY is selected so the relevant prior context can be resolved."
            )
        elif signals["multi_hop"]:
            strategy = "HYBRID"
            confidence = 0.9
            reason = (
                f"M3 identified a {routing_result.question_type} query requiring external retrieval, and the multi_hop signal is true. "
                "Because the answer may depend on evidence associated with different entities or relationships, HYBRID is selected to combine semantic and keyword-sensitive retrieval."
            )
        elif signals["keyword_sensitive_terms"] and signals["named_entity"]:
            strategy = "HYBRID"
            confidence = 0.90
            reason = (
                f"M3 identified a {routing_result.question_type} query requiring external retrieval. The keyword_sensitive_terms and named_entity "
                "signals are true, so exact lexical matching is combined with dense semantic embeddings via HYBRID retrieval."
            )
        else:
            strategy = "DENSE"
            confidence = 0.82
            reason = (
                f"M3 identified a {routing_result.question_type} query requiring external retrieval. The multi_hop signal is false and there is no "
                "combination of strong keyword-sensitive terms and a named entity, so a more specialized strategy is not indicated. "
                "Therefore, DENSE retrieval is selected because semantic similarity is suitable for conceptually relevant documents."
            )

        return RetrievalStrategyResult(
            query=routing_result.query,
            selected_strategy=strategy,
            route=routing_result.route,
            question_type=routing_result.question_type,
            requires_retrieval=requires_retrieval,
            requires_memory=requires_memory,
            confidence=confidence,
            reason=reason,
            signals=signals,
        )