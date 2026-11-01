import json

from modules.module3_adaptive_query_router.router import AdaptiveQueryRouter
from modules.module4_retrieval_strategy.schemas import RetrievalStrategyResult
from modules.module4_retrieval_strategy.strategy_selector import RetrievalStrategySelector
from modules.module5_retrieval.hybrid_retriever import HybridRetriever
from modules.module5_retrieval.calibration import DevelopmentCalibrator, HotPotQALabelStore, ScoreCalibrationModel, fit_score_model
from modules.module5_retrieval.evaluation import evaluate_development_retrieval
from modules.module5_retrieval.relevance_filter import ScoreBasedRelevanceFilter
from modules.module5_retrieval.retriever import RetrievalEngine
from modules.module5_retrieval.schemas import EvidenceDocument, RetrievalConfig
from modules.module5_retrieval.sparse_retriever import SparseRetriever
from config.config import HOTPOTQA_DOCS_OUTPUT, HOTPOTQA_QUESTIONS_OUTPUT


def strategy_for(question: str):
    routing = AdaptiveQueryRouter().route(question)
    return RetrievalStrategySelector().select(routing)


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def retrieve(self, query, top_k=None):
        self.calls += 1
        return [dict(item) for item in self.results[:top_k]]


def test_direct_returns_no_external_candidates():
    result = RetrievalEngine().retrieve(strategy_for("What is Python?"))
    assert result.strategy == "DIRECT"
    assert result.retrieval_executed is False
    assert result.candidate_count == 0
    assert result.relevant_count == 0
    assert result.documents == []


def test_memory_does_not_execute_external_retrieval():
    result = RetrievalEngine().retrieve(strategy_for("What about its advantages?"))
    assert result.strategy == "MEMORY"
    assert result.retrieval_executed is False
    assert result.documents == []


def test_sparse_bm25_returns_scores(tmp_path):
    path = tmp_path / "documents.jsonl"
    records = [
        {"doc_id": "a", "text": "Python virtual environment creation guide"},
        {"doc_id": "b", "text": "REST and SOAP comparison"},
        {"doc_id": "c", "text": "A short unrelated passage"},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    retriever = SparseRetriever(path, RetrievalConfig(retrieval_top_k=2, max_documents=3))
    results = retriever.retrieve("Python virtual environment", top_k=2)
    assert results
    assert results[0]["document_id"] == "a"
    assert all("sparse_score" in result for result in results)
    assert results[0]["retrieval_score"] >= results[1]["retrieval_score"]


def test_dense_strategy_executes_and_returns_evidence():
    fake_dense = FakeRetriever([{"doc_id": "d1", "text": "semantic result", "retrieval_score": 0.8}])
    result = RetrievalEngine(
        RetrievalConfig(retrieval_top_k=1, final_top_n=1, relevance_threshold=0.5),
        dense_retriever=fake_dense,
        sparse_retriever=FakeRetriever([]),
    ).retrieve(strategy_for("How do I create a virtual environment in Python?"))
    assert result.strategy == "DENSE"
    assert fake_dense.calls == 1
    assert result.candidate_count == 1
    assert result.documents[0].retrieval_score == 0.8
    assert result.documents[0].selected_for_evidence is True


def test_hybrid_fuses_both_retrievers_and_preserves_metadata():
    dense = FakeRetriever([{"doc_id": "a", "text": "dense text", "retrieval_score": 0.9}])
    sparse = FakeRetriever([{"doc_id": "a", "text": "sparse text", "retrieval_score": 4.0}])
    result = RetrievalEngine(
        RetrievalConfig(retrieval_top_k=1, final_top_n=1, relevance_threshold=0.5),
        dense_retriever=dense,
        sparse_retriever=sparse,
    ).retrieve(strategy_for("Which country was the birthplace of the person who founded the company?"))
    assert result.strategy == "HYBRID"
    assert dense.calls == 1
    assert sparse.calls == 1
    document = result.documents[0]
    assert document.retrieval_method == "HYBRID"
    assert document.dense_score == 0.9
    assert document.sparse_score == 4.0
    assert document.normalized_dense_score == 0.9
    assert document.normalized_sparse_score == 4.0
    assert document.fusion_score == 2.45
    assert document.selected_for_evidence is True


def test_relevance_filter_applies_threshold_and_top_n():
    candidates = [
        EvidenceDocument("a", "a", "DENSE", 1, 0.9, 0.9, False),
        EvidenceDocument("b", "b", "DENSE", 2, 0.7, 0.7, False),
        EvidenceDocument("c", "c", "DENSE", 3, 0.2, 0.2, False),
    ]
    filtered = ScoreBasedRelevanceFilter(threshold=0.5, final_top_n=1).apply(candidates)
    assert len(filtered) == 3
    assert sum(document.selected_for_evidence for document in filtered) == 1
    assert filtered[0].selected_for_evidence is True
    assert all(not document.selected_for_evidence for document in filtered[1:])


def test_development_calibration_selects_deterministic_f1_threshold(tmp_path):
    questions_path = tmp_path / "questions.jsonl"
    documents_path = tmp_path / "documents.jsonl"
    questions_path.write_text(
        json.dumps({"question_id": "q1", "question": "first question"}) + "\n"
        + json.dumps({"question_id": "q2", "question": "second question"}) + "\n",
        encoding="utf-8",
    )
    documents_path.write_text(
        "\n".join(json.dumps(record) for record in [
            {"doc_id": "q1-support", "question_id": "q1", "text": "support one", "metadata": {"is_supporting": True}},
            {"doc_id": "q1-other", "question_id": "q1", "text": "other one", "metadata": {"is_supporting": False}},
            {"doc_id": "q2-support", "question_id": "q2", "text": "support two", "metadata": {"is_supporting": True}},
            {"doc_id": "q2-other", "question_id": "q2", "text": "other two", "metadata": {"is_supporting": False}},
        ]),
        encoding="utf-8",
    )
    labels = HotPotQALabelStore(questions_path, documents_path, max_documents=4)
    scores = {
        "first question": [
            {"doc_id": "q1-support", "retrieval_score": 0.9, "normalized_score": 0.9},
            {"doc_id": "q1-other", "retrieval_score": 0.1, "normalized_score": 0.1},
        ],
        "second question": [
            {"doc_id": "q2-support", "retrieval_score": 0.8, "normalized_score": 0.8},
            {"doc_id": "q2-other", "retrieval_score": 0.2, "normalized_score": 0.2},
        ],
    }
    calibration = DevelopmentCalibrator(labels, query_limit=2, top_k=2).calibrate(
        "DENSE", lambda query, top_k: scores[query]
    )
    assert 0.0 < calibration.threshold < 1.0
    assert calibration.f1 > 0.5
    assert "maximum micro-F1" in calibration.objective


def test_supporting_document_evaluation_recognizes_hotpot_labels(tmp_path):
    questions_path = tmp_path / "questions.jsonl"
    documents_path = tmp_path / "documents.jsonl"
    questions_path.write_text(json.dumps({"question_id": "q1", "question": "Which magazine?"}) + "\n", encoding="utf-8")
    documents_path.write_text("\n".join(json.dumps(record) for record in [
        {"doc_id": "q1-support", "question_id": "q1", "text": "Arthur's Magazine", "metadata": {"is_supporting": True}},
        {"doc_id": "q1-other", "question_id": "q1", "text": "First for Women", "metadata": {"is_supporting": False}},
    ]), encoding="utf-8")
    labels = HotPotQALabelStore(questions_path, documents_path, max_documents=2)
    metrics = evaluate_development_retrieval(
        "SPARSE", labels,
        lambda query, top_k: [
            {"doc_id": "q1-support", "retrieval_score": 2.0, "normalized_score": 1.0},
            {"doc_id": "q1-other", "retrieval_score": 0.2, "normalized_score": 0.0},
        ],
        top_k=2,
        threshold=0.5,
    )
    assert metrics["supporting_documents_retrieved"] == 1
    assert metrics["supporting_document_recall"] == 1.0
    assert metrics["retained_supporting_document_precision"] == 1.0


def test_real_hotpotqa_development_supporting_label_is_recognized():
    labels = HotPotQALabelStore(HOTPOTQA_QUESTIONS_OUTPUT, HOTPOTQA_DOCS_OUTPUT, max_documents=5000)
    question = labels.calibration_queries(1)[0]
    question_id = str(question["question_id"])
    supporting_id = sorted(labels.supporting_document_ids[question_id])[0]
    metrics = evaluate_development_retrieval(
        "SPARSE",
        labels,
        lambda query, top_k: [{"doc_id": supporting_id, "retrieval_score": 1.0, "normalized_score": 1.0}],
        top_k=1,
        threshold=0.5,
        query_limit=1,
    )
    assert metrics["supporting_documents_retrieved"] == 1
    assert metrics["supporting_document_recall"] > 0.0


def test_exact_duplicate_candidate_is_removed():
    dense = FakeRetriever([
        {"doc_id": "a", "text": "same text", "retrieval_score": 0.9},
        {"doc_id": "b", "text": "  SAME   TEXT ", "retrieval_score": 0.8},
    ])
    result = RetrievalEngine(
        RetrievalConfig(retrieval_top_k=2, final_top_n=2, relevance_threshold=0.0),
        dense_retriever=dense,
    ).retrieve(strategy_for("How do I create a virtual environment in Python?"))
    assert len(result.documents) == 1
    assert result.deduplicated_count == 1
    assert result.duplicate_document_ids == ["b"]
