from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import pytest

from modules.caro_pipeline import CAROPipeline
from modules.module2_baseline_rag.embeddings import EmbeddingModel
from modules.module3_adaptive_query_router.router import AdaptiveQueryRouter
from modules.module4_retrieval_strategy.schemas import RetrievalStrategyResult
from modules.module4_retrieval_strategy.strategy_selector import RetrievalStrategySelector
from modules.module5_retrieval.calibration import DevelopmentCalibrator, HotPotQALabelStore
from modules.module5_retrieval.relevance_filter import ScoreBasedRelevanceFilter
from modules.module5_retrieval.retriever import RetrievalEngine
from modules.module5_retrieval.schemas import EvidenceDocument, RetrievalConfig
from modules.module6_temporary_memory.memory_manager import TemporaryMemoryManager
from modules.module7_evidence_sufficiency.evidence_sufficiency import EvidenceSufficiencyAnalyzer
from modules.module8_answer_generation.generator import GroundedAnswerGenerator
from modules.module8_answer_generation.validator import AnswerValidator


def test_routing_natural_knowledge_questions():
    """Verify that natural knowledge-seeking questions are NOT misrouted to DIRECT."""
    router = AdaptiveQueryRouter()

    # Factual knowledge questions must route to RETRIEVE
    assert router.route("Where is the Eiffel Tower located?").route == "RETRIEVE"
    assert router.route("Who built the Eiffel Tower?").route == "RETRIEVE"
    assert router.route("When was the Eiffel Tower constructed?").route == "RETRIEVE"
    assert router.route("What is the capital of France?").route == "RETRIEVE"

    # Multi-hop questions must route to RETRIEVE with multi-hop question_type
    res_multihop = router.route("What is the capital of the country where the Eiffel Tower is located?")
    assert res_multihop.route == "RETRIEVE"
    assert res_multihop.question_type == "multi-hop"

    # Comparison questions must route to RETRIEVE with comparison question_type
    res_comp = router.route("Which magazine was started first Arthur's Magazine or First for Women?")
    assert res_comp.route == "RETRIEVE"
    assert res_comp.question_type == "comparison"

    # Definitional query with no named entity / factual relation should route to DIRECT
    res_def = router.route("What is Python?")
    assert res_def.route == "DIRECT"
    assert res_def.question_type == "definitional"

    # Pure conversational follow-up without external comparison must route to MEMORY
    res_mem = router.route("What about its advantages?")
    assert res_mem.route == "MEMORY"

    # Conversational query with external comparison must route to RETRIEVE_AND_MEMORY
    res_both = router.route("Which magazine was started first, that magazine or First for Women?")
    assert res_both.route == "RETRIEVE_AND_MEMORY"


def test_controlled_eiffel_tower_multihop_scenario(tmp_path):
    """Scenario 1: Verify multi-hop pipeline on a controlled index containing the landmark passages."""
    query = "What is the capital of the country where the Eiffel Tower is located?"

    # Verify M3 routing
    router = AdaptiveQueryRouter()
    routing = router.route(query)
    assert routing.route == "RETRIEVE"
    assert routing.question_type == "multi-hop"

    # Verify M4 strategy selection
    strategy_selector = RetrievalStrategySelector()
    strategy = strategy_selector.select(routing)
    assert strategy.selected_strategy == "HYBRID"

    # Controlled evidence containing both hops
    evidence = [
        {"title": "Eiffel Tower", "text": "The Eiffel Tower is an iron lattice tower located in Paris, France.", "retrieval_score": 0.95, "normalized_score": 0.95},
        {"title": "France", "text": "France is a country in Western Europe. Its capital and largest city is Paris.", "retrieval_score": 0.92, "normalized_score": 0.92},
    ]

    # Verify M7 sufficiency recognizes both hops
    analyzer = EvidenceSufficiencyAnalyzer()
    sufficiency = analyzer.analyze(query, evidence)
    assert sufficiency.sufficient is True
    assert sufficiency.action == "ANSWER"

    # Verify M8 answer generation and validation
    gen = GroundedAnswerGenerator(model_name="dummy-model")
    ans_res = gen.generate(query, evidence)
    assert ans_res.is_grounded is True
    assert "Paris" in ans_res.final_answer


def test_corpus_limitation_eiffel_tower_absence_in_distractor_index():
    """Document and verify that the landmark 'Eiffel Tower' passage is absent from development index."""
    meta_path = Path("indexes/hotpotqa/hotpotqa_dev_index.meta.json")
    if meta_path.exists():
        docs = json.loads(meta_path.read_text(encoding="utf-8"))
        # Search for any document actually describing the Paris Eiffel Tower
        paris_eiffel_docs = [
            d for d in docs
            if "paris" in str(d.get("text", "")).lower() and "eiffel tower" in str(d.get("text", "")).lower()
        ]
        # Verified corpus limitation: zero documents about the Paris landmark exist in this index slice
        assert len(paris_eiffel_docs) == 0


def test_magazine_comparison_positive_scenario():
    """Scenario 2A: Verify chronological comparison when both passages with dates are supplied."""
    query = "Which magazine was started first Arthur's Magazine or First for Women?"
    evidence = [
        {
            "title": "Arthur's Magazine",
            "text": "Arthur's Magazine (1844-1846) was an American literary magazine published in Philadelphia in the 19th century.",
        },
        {
            "title": "First for Women",
            "text": "First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989.",
        },
    ]

    validator = AnswerValidator()
    # Test valid grounded comparative answers
    answer1 = "Arthur's Magazine was started first. Arthur's Magazine was founded in 1844, while First for Women began in 1989."
    val1 = validator.validate(answer1, evidence, query=query)
    assert val1.is_valid is True
    assert val1.is_grounded is True
    assert val1.status == "VALIDATED"

    answer2 = "Arthur's Magazine was started first (1844 vs. 1989)."
    val2 = validator.validate(answer2, evidence, query=query)
    assert val2.is_valid is True
    assert val2.is_grounded is True


def test_magazine_comparison_negative_abstention_scenario():
    """Scenario 2B: Negative test asserting abstention when evidence does not support dates."""
    query = "Which magazine was started first Arthur's Magazine or First for Women?"
    irrelevant_evidence = [
        {
            "title": "Unrelated Magazine",
            "text": "Popular Science is an American magazine carrying popular science articles.",
        }
    ]

    validator = AnswerValidator()
    # Explicit abstention must be accepted as grounded
    abstention_answer = "The provided evidence is insufficient to answer the query."
    val = validator.validate(abstention_answer, irrelevant_evidence, query=query)
    assert val.is_valid is True
    assert val.status == "ABSTAINED"

    # Hallucinated answer with unrelated evidence must be rejected
    hallucinated_answer = "Arthur's Magazine was started first in 1844."
    val_hallucinated = validator.validate(hallucinated_answer, irrelevant_evidence, query=query)
    assert val_hallucinated.is_valid is False
    assert val_hallucinated.status == "UNGROUNDED"


def test_memory_plus_retrieval_strategy_resolution():
    """Scenario 4: Verify MEMORY_AND_RETRIEVAL resolves external strategy properly."""
    engine = RetrievalEngine()

    # When query has multi-hop signals, external retrieval resolves to HYBRID
    strat_multihop = RetrievalStrategyResult(
        query="Which magazine was started first, that magazine or First for Women?",
        selected_strategy="MEMORY_AND_RETRIEVAL",
        route="RETRIEVE_AND_MEMORY",
        question_type="comparison",
        requires_retrieval=True,
        requires_memory=True,
        confidence=0.9,
        reason="test",
        signals={"multi_hop": True, "keyword_sensitive_terms": False, "named_entity": True},
    )
    assert engine._effective_strategy("MEMORY_AND_RETRIEVAL", strat_multihop.signals) == "HYBRID"

    # When query has keyword and entity signals, external retrieval resolves to SPARSE
    strat_sparse = RetrievalStrategyResult(
        query="What is its official release date?",
        selected_strategy="MEMORY_AND_RETRIEVAL",
        route="RETRIEVE_AND_MEMORY",
        question_type="factual",
        requires_retrieval=True,
        requires_memory=True,
        confidence=0.85,
        reason="test",
        signals={"multi_hop": False, "keyword_sensitive_terms": True, "named_entity": True},
    )
    assert engine._effective_strategy("MEMORY_AND_RETRIEVAL", strat_sparse.signals) == "SPARSE"

    # Default fallback resolves to DENSE
    strat_dense = RetrievalStrategyResult(
        query="Tell me about its context.",
        selected_strategy="MEMORY_AND_RETRIEVAL",
        route="RETRIEVE_AND_MEMORY",
        question_type="general",
        requires_retrieval=True,
        requires_memory=True,
        confidence=0.8,
        reason="test",
        signals={},
    )
    assert engine._effective_strategy("MEMORY_AND_RETRIEVAL", strat_dense.signals) == "DENSE"


def test_calibration_held_out_data_split(tmp_path):
    """Scenario 5: Verify calibration train/test splits are independent and disjoint."""
    questions_path = tmp_path / "questions.jsonl"
    documents_path = tmp_path / "documents.jsonl"

    questions = [
        {"question_id": f"q{i}", "question": f"Question {i}"}
        for i in range(10)
    ]
    questions_path.write_text("\n".join(json.dumps(q) for q in questions), encoding="utf-8")

    documents = [
        {"doc_id": f"doc_{i}", "question_id": f"q{i}", "text": f"Document text {i}", "metadata": {"is_supporting": True}}
        for i in range(10)
    ]
    documents_path.write_text("\n".join(json.dumps(d) for d in documents), encoding="utf-8")

    label_store = HotPotQALabelStore(questions_path, documents_path, max_documents=10)

    train_queries = label_store.calibration_queries(split="train", train_ratio=0.8)
    test_queries = label_store.calibration_queries(split="test", train_ratio=0.8)

    train_ids = {q["question_id"] for q in train_queries}
    test_ids = {q["question_id"] for q in test_queries}

    # Verify non-empty and completely disjoint splits
    assert len(train_ids) == 8
    assert len(test_ids) == 2
    assert train_ids.isdisjoint(test_ids)


def test_offline_dependency_failure_handling():
    """Scenario 6: Verify graceful handling when generation dependency fails."""
    validator = AnswerValidator()

    # Empty answer
    val_empty = validator.validate("", [])
    assert val_empty.is_valid is False
    assert val_empty.status == "EMPTY"

    # Generation error message
    val_err = validator.validate("Generation failed: Connection refused.", [])
    assert val_err.is_valid is False
    assert val_err.status == "MALFORMED"


def test_embedding_model_thread_safety():
    """Verify thread-safe singleton cache across concurrent workers."""
    def worker():
        model = EmbeddingModel()
        return id(model.model)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker) for _ in range(8)]
        results = [f.result() for f in futures]

    # All workers must share the exact same underlying model instance
    assert len(set(results)) == 1
