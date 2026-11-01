from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from config.config import DEFAULT_TOP_K, HOTPOTQA_DOCS_OUTPUT, HOTPOTQA_INDEX_PATH, HOTPOTQA_QUESTIONS_OUTPUT, MAX_INDEX_DOCUMENTS
from modules.module4_retrieval_strategy.schemas import RetrievalStrategyResult

from .calibration import DevelopmentCalibrator, HotPotQALabelStore, ScoreCalibrationModel
from .dense_retriever import DenseRetriever
from .hybrid_retriever import HybridRetriever
from .relevance_filter import ScoreBasedRelevanceFilter
from .schemas import EvidenceDocument, RetrievalConfig, RetrievalResult
from .sparse_retriever import SparseRetriever


class RetrievalEngine:
    """Module 5 entry point: execute M4's strategy and filter evidence."""

    def __init__(self, config: RetrievalConfig | None = None, *, dense_retriever=None, sparse_retriever=None):
        self.config = config or RetrievalConfig()
        self.config.retrieval_top_k = self.config.retrieval_top_k or DEFAULT_TOP_K
        if self.config.max_documents is None and getattr(self.config, "index_type", "dev") == "dev":
            self.config.max_documents = MAX_INDEX_DOCUMENTS
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

    def retrieve(self, strategy_result: RetrievalStrategyResult) -> RetrievalResult:
        if not isinstance(strategy_result, RetrievalStrategyResult):
            raise TypeError("strategy_result must be a Module 4 RetrievalStrategyResult")
        strategy = strategy_result.selected_strategy
        if strategy in {"DIRECT", "MEMORY"}:
            message = (
                "No external retrieval was required by Module 4."
                if strategy == "DIRECT"
                else "External retrieval was not executed because Module 4 selected a memory-dependent strategy."
            )
            return RetrievalResult(
                query=strategy_result.query,
                strategy=strategy,
                candidate_count=0,
                relevant_count=0,
                unique_candidate_count=0,
                documents=[],
                retrieval_executed=False,
                message=message,
                config=asdict(self.config),
            )

        effective_strategy = self._effective_strategy(strategy, getattr(strategy_result, "signals", None))
        threshold, calibration = self._filter_configuration(effective_strategy)
        candidates = self._retrieve_external(effective_strategy, strategy_result.query, calibration)
        if self.config.deduplicate_candidates:
            unique_candidates, duplicate_document_ids = self._deduplicate(candidates)
        else:
            unique_candidates, duplicate_document_ids = candidates, []
        evidence_candidates = [self._to_evidence(document, effective_strategy, calibration) for document in unique_candidates]
        filtered = ScoreBasedRelevanceFilter(
            threshold=threshold,
            final_top_n=self.config.final_top_n,
        ).apply(evidence_candidates)
        return RetrievalResult(
            query=strategy_result.query,
            strategy=strategy,
            candidate_count=len(candidates),
            relevant_count=sum(document.selected_for_evidence for document in filtered),
            unique_candidate_count=len(filtered),
            documents=filtered,
            retrieval_executed=True,
            message="Retrieved candidates were deduplicated by exact normalized content and score-based relevance filtering was applied.",
            config=asdict(self.config),
            deduplicated_count=len(duplicate_document_ids),
            duplicate_document_ids=duplicate_document_ids,
            calibration=calibration,
        )

    def _filter_configuration(self, strategy: str) -> tuple[float, dict[str, Any]]:
        calibration_strategy = "DENSE" if strategy == "MEMORY_AND_RETRIEVAL" else strategy
        if self.config.relevance_threshold is not None:
            return float(self.config.relevance_threshold), {
                "mode": "explicit",
                "strategy": calibration_strategy,
                "threshold": float(self.config.relevance_threshold),
            }
        artifact_path = Path(self.config.calibration_path or (Path(HOTPOTQA_INDEX_PATH).parent / "m5_calibration.json"))
        artifact = {}
        if artifact_path.exists():
            try:
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                artifact = {}
        if calibration_strategy in artifact:
            result = artifact[calibration_strategy]
            return float(result["threshold"]), {"mode": "development_calibrated", **result}
        label_store = HotPotQALabelStore(
            HOTPOTQA_QUESTIONS_OUTPUT,
            self.config.documents_path or HOTPOTQA_DOCS_OUTPUT,
            self.config.max_documents,
        )
        calibrator = DevelopmentCalibrator(label_store, self.config.calibration_query_limit, self.config.calibration_top_k)
        if calibration_strategy == "SPARSE":
            result = calibrator.calibrate(strategy, lambda query, top_k: self._get_sparse().retrieve(query, top_k))
        elif calibration_strategy == "DENSE":
            result = calibrator.calibrate(strategy, lambda query, top_k: self._get_dense().retrieve(query, top_k))
        elif calibration_strategy == "HYBRID":
            result = calibrator.calibrate_hybrid(
                lambda query, top_k: self._get_dense().retrieve(query, top_k),
                lambda query, top_k: self._get_sparse().retrieve(query, top_k),
                self.config.dense_weight,
                self.config.sparse_weight,
            )
        else:
            raise ValueError(f"No calibration implementation for strategy: {strategy}")
        artifact[calibration_strategy] = result.to_dict()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        return result.threshold, {"mode": "development_calibrated", **result.to_dict()}

    @staticmethod
    def _deduplicate(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        seen: dict[str, dict[str, Any]] = {}
        duplicate_document_ids: list[str] = []
        for candidate in candidates:
            text = str(candidate.get("text") or candidate.get("content") or candidate.get("passage") or "")
            normalized = re.sub(r"\s+", " ", text).strip().casefold()
            key = hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else str(candidate.get("document_id") or candidate.get("doc_id") or candidate.get("id"))
            document_id = str(candidate.get("document_id") or candidate.get("doc_id") or candidate.get("id") or "")
            if key in seen:
                duplicate_document_ids.append(document_id)
                continue
            seen[key] = candidate
        return list(seen.values()), duplicate_document_ids

    @staticmethod
    def _effective_strategy(strategy: str, signals: dict[str, bool] | None = None) -> str:
        if strategy != "MEMORY_AND_RETRIEVAL":
            return strategy
        if signals:
            if signals.get("multi_hop"):
                return "HYBRID"
            if signals.get("keyword_sensitive_terms") and signals.get("named_entity"):
                return "SPARSE"
        return "DENSE"

    def _retrieve_external(self, strategy: str, query: str, calibration: dict[str, Any]) -> list[dict[str, Any]]:
        if strategy == "DENSE":
            return self._get_dense().retrieve(query, self.config.retrieval_top_k)
        if strategy == "SPARSE":
            return self._get_sparse().retrieve(query, self.config.retrieval_top_k)
        if strategy == "HYBRID":
            return HybridRetriever(
                self._get_dense(),
                self._get_sparse(),
                self.config,
                self._hybrid_model(calibration, "dense"),
                self._hybrid_model(calibration, "sparse"),
            ).retrieve(query, self.config.retrieval_top_k)
        if strategy == "MEMORY_AND_RETRIEVAL":
            return self._get_dense().retrieve(query, self.config.retrieval_top_k)
        raise ValueError(f"Unsupported Module 4 strategy: {strategy}")

    def _get_dense(self):
        if self.dense_retriever is None:
            if self.config.index_path:
                index_path = self.config.index_path
            elif getattr(self.config, "index_type", "dev") == "full":
                if getattr(self.config, "dataset", "hotpotqa") == "qrecc":
                    from config.config import QRECC_FULL_INDEX_PATH
                    index_path = str(QRECC_FULL_INDEX_PATH)
                else:
                    from config.config import HOTPOTQA_FULL_INDEX_PATH
                    index_path = str(HOTPOTQA_FULL_INDEX_PATH)
            else:
                index_path = str(HOTPOTQA_INDEX_PATH)
            self.dense_retriever = DenseRetriever(index_path, self.config)
        return self.dense_retriever

    def _get_sparse(self):
        if self.sparse_retriever is None:
            if self.config.documents_path:
                documents_path = self.config.documents_path
            elif getattr(self.config, "dataset", "hotpotqa") == "qrecc":
                from config.config import QRECC_DOCS_OUTPUT
                documents_path = str(QRECC_DOCS_OUTPUT)
            else:
                documents_path = str(HOTPOTQA_DOCS_OUTPUT)
            self.sparse_retriever = SparseRetriever(documents_path, self.config)
        return self.sparse_retriever

    @staticmethod
    def _to_evidence(document: dict[str, Any], strategy: str, calibration: dict[str, Any]) -> EvidenceDocument:
        method = "HYBRID" if strategy == "HYBRID" else "DENSE" if strategy in {"DENSE", "MEMORY_AND_RETRIEVAL"} else "SPARSE"
        score = float(document.get("retrieval_score", 0.0))
        normalized_score = float(document.get("fusion_score", document.get("normalized_score", score)))
        # --- BUGGED ORIGINAL (Uncomment to revert) ---
        # if strategy in {"DENSE", "SPARSE"} and calibration.get("score_model"):
        #     model = ScoreCalibrationModel(**calibration["score_model"])
        #     normalized_score = model.transform(score)
        # --- END BUGGED ORIGINAL ---

        # --- FIX: Include MEMORY_AND_RETRIEVAL since it uses DENSE calibration model ---
        if strategy in {"DENSE", "SPARSE", "MEMORY_AND_RETRIEVAL"} and calibration.get("score_model"):
            model = ScoreCalibrationModel(**calibration["score_model"])
            normalized_score = model.transform(score)
        # --- END FIX ---
        metadata = dict(document.get("metadata")) if isinstance(document.get("metadata"), dict) else {}
        for key in ("title", "source", "split", "question_id"):
            if document.get(key) is not None and key not in metadata:
                metadata[key] = document[key]
        return EvidenceDocument(
            document_id=str(document.get("document_id") or document.get("doc_id") or document.get("id") or ""),
            text=str(document.get("text") or document.get("content") or document.get("passage") or ""),
            retrieval_method=method,
            retrieval_rank=int(document.get("retrieval_rank", 0)),
            retrieval_score=score,
            normalized_score=normalized_score,
            selected_for_evidence=False,
            filter_reason="pending",
            metadata=metadata,
            dense_score=document.get("dense_score"),
            sparse_score=document.get("sparse_score"),
            normalized_dense_score=document.get("normalized_dense_score"),
            normalized_sparse_score=document.get("normalized_sparse_score"),
            fusion_score=document.get("fusion_score"),
        )

    @staticmethod
    def _hybrid_model(calibration: dict[str, Any], component: str):
        model = calibration.get("score_model", {})
        prefix = f"{component}_"
        fields = {
            key[len(prefix):]: value
            for key, value in model.items()
            if key.startswith(prefix) and key[len(prefix):] in {"mean", "scale", "slope", "intercept"}
        }
        return ScoreCalibrationModel(**fields) if fields else None
