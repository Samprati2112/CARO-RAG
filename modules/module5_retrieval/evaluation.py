from __future__ import annotations

from typing import Any, Callable

from .calibration import HotPotQALabelStore, ScoreCalibrationModel


def evaluate_development_retrieval(
    strategy: str,
    label_store: HotPotQALabelStore,
    retrieve: Callable[[str, int], list[dict[str, Any]]],
    *,
    top_k: int = 5,
    threshold: float | None = None,
    query_limit: int | None = None,
    score_model: ScoreCalibrationModel | None = None,
    split: str = "all",
) -> dict[str, Any]:
    """Measure retrieval and score-based retention against HotPotQA labels."""

    query_results = []
    retrieved_count = 0
    retained_count = 0
    supporting_retrieved = 0
    supporting_retained = 0
    supporting_available = 0
    duplicate_count = 0
    for question in label_store.calibration_queries(query_limit, split=split):
        question_id = str(question["question_id"])
        support_ids = label_store.supporting_document_ids.get(question_id, set())
        supporting_available += len(support_ids)
        candidates = retrieve(str(question["question"]), top_k)
        seen_text: set[str] = set()
        unique_candidates = []
        for candidate in candidates:
            text = " ".join(str(candidate.get("text", "")).split()).casefold()
            if text and text in seen_text:
                duplicate_count += 1
                continue
            seen_text.add(text)
            unique_candidates.append(candidate)
        retrieved_count += len(unique_candidates)
        query_supporting_retrieved = sum(label_store.is_supporting(question_id, candidate) for candidate in unique_candidates)
        supporting_retrieved += query_supporting_retrieved
        retained = [
            candidate
            for candidate in unique_candidates
            if threshold is None or _score(strategy, candidate, score_model) >= threshold
        ]
        retained_count += len(retained)
        supporting_retained += sum(label_store.is_supporting(question_id, candidate) for candidate in retained)
        query_results.append({
            "question_id": question_id,
            "retrieved_count": len(unique_candidates),
            "retained_count": len(retained),
            "supporting_retrieved": query_supporting_retrieved,
        })
    precision = supporting_retained / retained_count if retained_count else 0.0
    recall = supporting_retrieved / supporting_available if supporting_available else 0.0
    return {
        "strategy": strategy,
        "query_count": len(query_results),
        "retrieved_candidates": retrieved_count,
        "retained_candidates": retained_count,
        "percentage_filtered": 100.0 * (retrieved_count - retained_count) / retrieved_count if retrieved_count else 0.0,
        "supporting_documents_available": supporting_available,
        "supporting_documents_retrieved": supporting_retrieved,
        "supporting_document_recall": recall,
        "retained_supporting_document_precision": precision,
        "duplicate_count": duplicate_count,
        "queries": query_results,
    }


def _score(strategy: str, candidate: dict[str, Any], score_model: ScoreCalibrationModel | None = None) -> float:
    if strategy == "HYBRID":
        return float(candidate.get("fusion_score", candidate.get("retrieval_score", 0.0)))
    if score_model is not None:
        return score_model.transform(float(candidate.get("retrieval_score", 0.0)))
    return float(candidate.get("normalized_score", candidate.get("retrieval_score", 0.0)))
