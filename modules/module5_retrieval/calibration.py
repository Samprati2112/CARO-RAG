from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .schemas import RetrievalConfig


@dataclass
class CalibrationResult:
    """Deterministic development-data calibration artifact for one strategy."""

    strategy: str
    threshold: float
    objective: str
    calibration_query_count: int
    calibration_candidate_count: int
    positive_candidate_count: int
    selected_candidate_count: int
    precision: float
    recall: float
    f1: float
    data_source: str
    corpus_limit: int
    score_model: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HotPotQALabelStore:
    """Read supporting-document labels from standardized HotPotQA outputs."""

    def __init__(self, questions_path: str | Path, documents_path: str | Path, max_documents: int = 5000):
        self.questions_path = Path(questions_path)
        self.documents_path = Path(documents_path)
        self.max_documents = max_documents
        self.questions: list[dict[str, Any]] = []
        self.supporting_document_ids: dict[str, set[str]] = {}
        self._load()

    def _load(self) -> None:
        documents_by_question: dict[str, list[dict[str, Any]]] = {}
        with self.documents_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                document = json.loads(line)
                question_id = str(document.get("question_id", ""))
                documents_by_question.setdefault(question_id, []).append(document)
                if sum(len(items) for items in documents_by_question.values()) >= self.max_documents:
                    break
        with self.questions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                question = json.loads(line)
                question_id = str(question.get("question_id", ""))
                support_ids = {
                    str(document.get("doc_id"))
                    for document in documents_by_question.get(question_id, [])
                    if isinstance(document.get("metadata"), dict) and document["metadata"].get("is_supporting") is True
                }
                if support_ids:
                    self.questions.append(question)
                    self.supporting_document_ids[question_id] = support_ids

    def calibration_queries(
        self,
        limit: int | None = None,
        split: str = "all",
        train_ratio: float = 0.8,
    ) -> list[dict[str, Any]]:
        queries = sorted(self.questions, key=lambda item: str(item.get("question_id", "")))
        if split == "train" and len(queries) > 1:
            split_idx = max(1, int(len(queries) * train_ratio))
            queries = queries[:split_idx]
        elif split == "test" and len(queries) > 1:
            split_idx = max(1, int(len(queries) * train_ratio))
            queries = queries[split_idx:]
        return queries[:limit] if limit is not None else queries

    def is_supporting(self, question_id: str, document: dict[str, Any]) -> bool:
        document_id = str(document.get("document_id") or document.get("doc_id") or document.get("id") or "")
        return document_id in self.supporting_document_ids.get(str(question_id), set())


class DevelopmentCalibrator:
    """Calibrate score-based filtering from labeled development candidates only."""

    def __init__(self, label_store: HotPotQALabelStore, query_limit: int = 128, top_k: int = 10):
        self.label_store = label_store
        self.query_limit = query_limit
        self.top_k = top_k

    def calibrate(self, strategy: str, retrieve: Callable[[str, int], list[dict[str, Any]]], split: str = "all") -> CalibrationResult:
        raw_points: list[tuple[float, bool]] = []
        query_count = 0
        for question in self.label_store.calibration_queries(self.query_limit, split=split):
            query_count += 1
            query_text = str(question.get("question", ""))
            candidates = retrieve(query_text, self.top_k)
            for candidate in candidates:
                score = self._raw_score(strategy, candidate)
                raw_points.append((score, self.label_store.is_supporting(str(question["question_id"]), candidate)))
        if not raw_points:
            raise ValueError("No labeled retrieval candidates were available for calibration.")
        score_model = fit_score_model(raw_points)
        points = [(score_model.transform(score), label) for score, label in raw_points]
        threshold, metrics = self._select_threshold(points)
        return CalibrationResult(
            strategy=strategy,
            threshold=threshold,
            objective="maximum micro-F1 over retrieved candidates; ties choose the highest threshold",
            calibration_query_count=query_count,
            calibration_candidate_count=len(points),
            positive_candidate_count=sum(label for _, label in points),
            selected_candidate_count=metrics["selected"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            data_source=str(self.label_store.documents_path),
            corpus_limit=self.label_store.max_documents,
            score_model=score_model.to_dict(),
        )

    def calibrate_hybrid(
        self,
        dense_retrieve: Callable[[str, int], list[dict[str, Any]]],
        sparse_retrieve: Callable[[str, int], list[dict[str, Any]]],
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        split: str = "all",
    ) -> CalibrationResult:
        dense_points: list[tuple[float, bool]] = []
        sparse_points: list[tuple[float, bool]] = []
        query_count = 0
        candidate_count = 0
        query_results: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
        for question in self.label_store.calibration_queries(self.query_limit, split=split):
            query_count += 1
            question_id = str(question["question_id"])
            query = str(question.get("question", ""))
            dense = dense_retrieve(query, self.top_k)
            sparse = sparse_retrieve(query, self.top_k)
            query_results.append((question_id, dense, sparse))
            for candidate in dense:
                dense_points.append((self._raw_score("DENSE", candidate), self.label_store.is_supporting(question_id, candidate)))
            for candidate in sparse:
                sparse_points.append((self._raw_score("SPARSE", candidate), self.label_store.is_supporting(question_id, candidate)))
            candidate_count += len({self._candidate_id(item) for item in dense + sparse})
        dense_model = fit_score_model(dense_points)
        sparse_model = fit_score_model(sparse_points)
        fused_points: list[tuple[float, bool]] = []
        for question_id, dense, sparse in query_results:
            dense_by_id = {self._candidate_id(item): item for item in dense}
            sparse_by_id = {self._candidate_id(item): item for item in sparse}
            for document_id in set(dense_by_id) | set(sparse_by_id):
                if document_id in dense_by_id and document_id in sparse_by_id:
                    score = (
                        dense_weight * dense_model.transform(self._raw_score("DENSE", dense_by_id[document_id]))
                        + sparse_weight * sparse_model.transform(self._raw_score("SPARSE", sparse_by_id[document_id]))
                    )
                elif document_id in dense_by_id:
                    score = dense_model.transform(self._raw_score("DENSE", dense_by_id[document_id]))
                else:
                    score = sparse_model.transform(self._raw_score("SPARSE", sparse_by_id[document_id]))
                candidate = dense_by_id.get(document_id, sparse_by_id.get(document_id, {}))
                fused_points.append((score, self.label_store.is_supporting(question_id, candidate)))
        threshold, metrics = self._select_threshold(fused_points)
        return CalibrationResult(
            strategy="HYBRID",
            threshold=threshold,
            objective="maximum micro-F1 over pooled calibrated fusion scores; ties choose the highest threshold",
            calibration_query_count=query_count,
            calibration_candidate_count=candidate_count,
            positive_candidate_count=sum(label for _, label in fused_points),
            selected_candidate_count=metrics["selected"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            data_source=str(self.label_store.documents_path),
            corpus_limit=self.label_store.max_documents,
            score_model={
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "dense_mean": dense_model.mean,
                "dense_scale": dense_model.scale,
                "dense_slope": dense_model.slope,
                "dense_intercept": dense_model.intercept,
                "sparse_mean": sparse_model.mean,
                "sparse_scale": sparse_model.scale,
                "sparse_slope": sparse_model.slope,
                "sparse_intercept": sparse_model.intercept,
            },
        )

    @staticmethod
    def _raw_score(strategy: str, candidate: dict[str, Any]) -> float:
        return float(candidate.get("retrieval_score", candidate.get("dense_score", candidate.get("sparse_score", 0.0))))

    @staticmethod
    def _candidate_id(candidate: dict[str, Any]) -> str:
        return str(candidate.get("document_id") or candidate.get("doc_id") or candidate.get("id") or "")

    @staticmethod
    def _select_threshold(points: list[tuple[float, bool]]) -> tuple[float, dict[str, float]]:
        thresholds = sorted({score for score, _ in points}, reverse=True)
        best: tuple[float, dict[str, float]] | None = None
        for threshold in thresholds:
            selected = [label for score, label in points if score >= threshold]
            true_positive = sum(selected)
            predicted = len(selected)
            actual = sum(label for _, label in points)
            precision = true_positive / predicted if predicted else 0.0
            recall = true_positive / actual if actual else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            metrics = {"selected": predicted, "precision": precision, "recall": recall, "f1": f1}
            if best is None or (f1, threshold, precision, recall) > (best[1]["f1"], best[0], best[1]["precision"], best[1]["recall"]):
                best = (threshold, metrics)
        assert best is not None
        return best


def calibration_cache_key(config: RetrievalConfig, strategy: str) -> str:
    payload = f"{strategy}|{config.max_documents}|{config.retrieval_top_k}|{config.dense_weight}|{config.sparse_weight}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class ScoreCalibrationModel:
    """Global raw-score-to-calibrated-score transform fitted on development data."""

    mean: float
    scale: float
    slope: float
    intercept: float

    def transform(self, score: float) -> float:
        value = self.slope * ((float(score) - self.mean) / self.scale) + self.intercept
        value = max(-60.0, min(60.0, value))
        import math
        return 1.0 / (1.0 + math.exp(-value))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_score_model(points: list[tuple[float, bool]]) -> ScoreCalibrationModel:
    """Fit a deterministic regularized one-feature logistic calibration model."""
    scores = [float(score) for score, _ in points]
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    scale = variance ** 0.5 or 1.0
    slope, intercept = 1.0, 0.0
    regularization = 1e-3
    for _ in range(40):
        gradient_slope = -regularization * slope
        gradient_intercept = -regularization * intercept
        hessian_ss = regularization
        hessian_si = 0.0
        hessian_ii = regularization
        for score, label in points:
            feature = (float(score) - mean) / scale
            linear = max(-60.0, min(60.0, slope * feature + intercept))
            import math
            probability = 1.0 / (1.0 + math.exp(-linear))
            error = float(label) - probability
            gradient_slope += error * feature
            gradient_intercept += error
            weight = probability * (1.0 - probability)
            hessian_ss += weight * feature * feature
            hessian_si += weight * feature
            hessian_ii += weight
        determinant = hessian_ss * hessian_ii - hessian_si * hessian_si
        if determinant == 0.0:
            break
        delta_slope = (gradient_slope * hessian_ii - gradient_intercept * hessian_si) / determinant
        delta_intercept = (gradient_intercept * hessian_ss - gradient_slope * hessian_si) / determinant
        slope += delta_slope
        intercept += delta_intercept
        if abs(delta_slope) + abs(delta_intercept) < 1e-8:
            break
    return ScoreCalibrationModel(mean, scale, slope, intercept)
