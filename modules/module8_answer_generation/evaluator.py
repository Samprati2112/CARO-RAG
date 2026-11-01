from __future__ import annotations

import re
import string
from typing import Any, Sequence

from .schemas import EvaluationMetrics


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class GenerationEvaluator:
    """Evaluates generation quality and evidence grounding."""

    @staticmethod
    def compute_exact_match(prediction: str, ground_truth: str) -> float:
        return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

    @staticmethod
    def compute_f1(prediction: str, ground_truth: str) -> float:
        pred_tokens = normalize_answer(prediction).split()
        truth_tokens = normalize_answer(ground_truth).split()

        if not pred_tokens or not truth_tokens:
            return 1.0 if pred_tokens == truth_tokens else 0.0

        common_tokens = set(pred_tokens) & set(truth_tokens)
        if not common_tokens:
            return 0.0

        # Token overlap counts
        num_same = sum(min(pred_tokens.count(tok), truth_tokens.count(tok)) for tok in common_tokens)
        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return round(f1, 4)

    @classmethod
    def compute_grounding_ratio(cls, prediction: str, evidence_texts: list[str]) -> float:
        if not prediction.strip() or not evidence_texts:
            return 0.0
        combined = " ".join(evidence_texts).casefold()
        tokens = [t for t in normalize_answer(prediction).split() if len(t) > 2]
        if not tokens:
            return 1.0
        grounded_count = sum(1 for t in tokens if t in combined)
        return round(grounded_count / len(tokens), 4)

    @classmethod
    def evaluate(
        cls,
        prediction: str,
        ground_truth: str | None = None,
        evidence_texts: list[str] | None = None,
        query: str = "",
    ) -> EvaluationMetrics:
        em = 0.0
        f1 = 0.0
        if ground_truth is not None:
            em = cls.compute_exact_match(prediction, ground_truth)
            f1 = cls.compute_f1(prediction, ground_truth)

        grounding = cls.compute_grounding_ratio(prediction, evidence_texts or []) if evidence_texts else 0.0
        
        # Context relevance heuristic
        context_relevance = 0.0
        if query and evidence_texts:
            query_tokens = [t for t in normalize_answer(query).split() if len(t) > 2]
            if query_tokens:
                combined = " ".join(evidence_texts).casefold()
                covered = sum(1 for t in query_tokens if t in combined)
                context_relevance = round(covered / len(query_tokens), 4)

        return EvaluationMetrics(
            exact_match=em,
            f1_score=f1,
            grounding_ratio=grounding,
            context_relevance=context_relevance,
        )
