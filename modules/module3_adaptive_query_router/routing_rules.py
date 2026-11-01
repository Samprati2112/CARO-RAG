from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouteDecision:
    """Rule result for a candidate route."""

    route: str
    confidence: float
    reason: str


class RoutingRuleSet:
    """Heuristic routing rules for AQR.

    These rules are intentionally explicit and deterministic. The router may later be
    extended with more sophisticated heuristics without changing the public API.
    """

    @staticmethod
    def decide(question_type: str, analysis) -> RouteDecision:
        requires_memory = analysis.contains_follow_up_reference or analysis.contains_contextual_marker
        requires_retrieval = (
            analysis.has_procedural_signal
            or analysis.has_comparison_signal
            or analysis.has_multi_hop_signal
            or analysis.likely_needs_retrieval
            or question_type in {"factual", "multi-hop", "comparison", "procedural"}
        )

        detected_signals = [
            name
            for name, detected in (
                ("procedural", analysis.has_procedural_signal),
                ("comparison", analysis.has_comparison_signal),
                ("multi-hop", analysis.has_multi_hop_signal),
                ("contextual reference", analysis.contains_contextual_marker),
            )
            if detected
        ]
        signal_summary = ", ".join(detected_signals) if detected_signals else "no specialized procedural, comparison, multi-hop, or contextual-reference signal"
        characteristics = f"The query was classified as {question_type}; detected signals include {signal_summary}"
        memory_reason = (
            "a conversational dependency was detected from follow-up or contextual references"
            if requires_memory
            else "no conversational dependency on previous context was detected"
        )
        retrieval_reason = (
            "an explicit retrieval-related characteristic was detected under the current routing rules"
            if requires_retrieval
            else "no explicit retrieval requirement was detected under the current routing rules"
        )

        if requires_retrieval and requires_memory:
            reason = f"{characteristics}. {memory_reason.capitalize()}, and {retrieval_reason}. Therefore, CARO selects the RETRIEVE_AND_MEMORY route."
            return RouteDecision("RETRIEVE_AND_MEMORY", 0.82, reason)
        if requires_retrieval:
            reason = f"{characteristics}; {retrieval_reason}, while {memory_reason}. Therefore, CARO selects the RETRIEVE route."
            return RouteDecision("RETRIEVE", 0.8, reason)
        if requires_memory:
            reason = f"{characteristics}; {memory_reason}, while {retrieval_reason}. Therefore, CARO selects the MEMORY route."
            return RouteDecision("MEMORY", 0.8, reason)
        if analysis.likely_direct_answer:
            reason = f"{characteristics} and is simple and self-contained. {memory_reason.capitalize()}, and {retrieval_reason}. Therefore, CARO selects the DIRECT route."
            return RouteDecision("DIRECT", 0.7, reason)
        if question_type == "unknown":
            reason = f"{characteristics} and is unclear. {memory_reason.capitalize()}, and {retrieval_reason}. Therefore, CARO selects the DIRECT route pending more context."
            return RouteDecision("DIRECT", 0.45, reason)
        reason = f"{characteristics}. {memory_reason.capitalize()}, and {retrieval_reason}. Therefore, CARO selects the DIRECT route under the current routing rules."
        return RouteDecision("DIRECT", 0.6, reason)
