import json

from modules.module5_retrieval.calibration import ScoreCalibrationModel, fit_score_model
from modules.module5_retrieval.hybrid_retriever import HybridRetriever
from modules.module5_retrieval.retriever import RetrievalEngine
from modules.module5_retrieval.schemas import RetrievalConfig
from modules.module5_retrieval.sparse_retriever import SparseRetriever
from modules.module3_adaptive_query_router.router import AdaptiveQueryRouter
from modules.module4_retrieval_strategy.schemas import RetrievalStrategyResult


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k=None):
        return self.results[:top_k]


def test_global_score_calibration_does_not_map_each_query_top_candidate_to_one():
    model = fit_score_model([(0.2, False), (0.4, False), (0.8, True), (1.0, True)])
    first = model.transform(0.8)
    second = model.transform(0.4)
    assert 0.0 < first < 1.0
    assert 0.0 < second < 1.0
    assert first != 1.0
    assert second != 1.0
    assert first > second


def test_hybrid_calibrated_components_keep_complementary_supporting_candidates():
    dense_model = ScoreCalibrationModel(mean=0.5, scale=0.2, slope=1.0, intercept=0.0)
    sparse_model = ScoreCalibrationModel(mean=0.5, scale=0.2, slope=1.0, intercept=0.0)
    dense = FakeRetriever([
        {"doc_id": "blackberry", "text": "BlackBerry Limited", "retrieval_score": 0.9},
        {"doc_id": "huddersfield", "text": "Huddersfield Giants", "retrieval_score": 0.4},
    ])
    sparse = FakeRetriever([
        {"doc_id": "blackberry", "text": "BlackBerry Limited", "retrieval_score": 0.4},
        {"doc_id": "huddersfield", "text": "Huddersfield Giants", "retrieval_score": 0.9},
    ])
    fused = HybridRetriever(dense, sparse, RetrievalConfig(), dense_model, sparse_model).retrieve("multi-hop", 2)
    assert all(item["fusion_score"] > 0.5 for item in fused)


def test_sparse_scores_are_not_query_local_min_max_scores(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in [
        {"doc_id": "a", "text": "population population"},
        {"doc_id": "b", "text": "population"},
    ]), encoding="utf-8")
    results = SparseRetriever(path, RetrievalConfig(retrieval_top_k=1, max_documents=2)).retrieve("population", 1)
    assert results[0]["normalized_score"] == results[0]["retrieval_score"]


def test_persisted_calibration_threshold_is_reused_across_queries(tmp_path):
    artifact = tmp_path / "calibration.json"
    artifact.write_text(
        json.dumps({
            "SPARSE": {
                "strategy": "SPARSE",
                "threshold": 0.7,
                "score_model": {"mean": 0.0, "scale": 1.0, "slope": 1.0, "intercept": 0.0},
            }
        }),
        encoding="utf-8",
    )
    class FakeSparse:
        def retrieve(self, query, top_k=None):
            return [{"doc_id": query, "text": query, "retrieval_score": 1.0 if query == "q1" else 0.5}]

    engine = RetrievalEngine(
        RetrievalConfig(retrieval_top_k=1, final_top_n=1, calibration_path=str(artifact)),
        sparse_retriever=FakeSparse(),
    )
    def strategy(query):
        return RetrievalStrategyResult(
            query=query,
            selected_strategy="SPARSE",
            route="RETRIEVE",
            question_type="factual",
            requires_retrieval=True,
            requires_memory=False,
            confidence=0.8,
            reason="test",
            signals={},
        )

    first = engine.retrieve(strategy("q1"))
    second = engine.retrieve(strategy("q2"))
    assert first.calibration["threshold"] == 0.7
    assert second.calibration["threshold"] == 0.7
    assert first.relevant_count == 1
    assert second.relevant_count == 0
