"""Adaptive Query Router (AQR) module for CARO.

This module provides a deterministic, explainable routing layer that analyzes an
incoming user query and decides whether it should be handled as a direct answer,
external retrieval, memory-dependent request, or both.
"""

from .router import AdaptiveQueryRouter
from .query_analyzer import QueryAnalysis, QueryAnalyzer
from .routing_rules import RouteDecision, RoutingRuleSet
from .schemas import RoutingResult

__all__ = [
    "AdaptiveQueryRouter",
    "QueryAnalysis",
    "QueryAnalyzer",
    "RouteDecision",
    "RoutingRuleSet",
    "RoutingResult",
]
