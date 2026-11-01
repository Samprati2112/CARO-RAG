from __future__ import annotations

from collections.abc import Iterable

from .schemas import EvidenceDocument


class ScoreBasedRelevanceFilter:
    """Apply configurable score thresholding and final top-N selection."""

    def __init__(self, threshold: float = 0.0, final_top_n: int = 5):
        if final_top_n < 0:
            raise ValueError("final_top_n must be non-negative")
        self.threshold = float(threshold)
        self.final_top_n = final_top_n

    def apply(self, candidates: Iterable[EvidenceDocument]) -> list[EvidenceDocument]:
        ranked = sorted(candidates, key=lambda item: item.normalized_score, reverse=True)
        selected_ids = {
            id(document)
            for document in ranked[: self.final_top_n]
            if document.normalized_score >= self.threshold
        }
        for document in ranked:
            document.selected_for_evidence = id(document) in selected_ids
            document.filter_reason = (
                "retained_score_threshold"
                if document.selected_for_evidence
                else "removed_relevance_threshold"
            )
        return ranked
