"""
Comprehensive unit tests for the fair evaluation harness.
Verifies:
1. Answer normalization, EM, and Token F1 on known HotpotQA examples
2. Retrieval Recall@k, Precision@k, and multi-hop success on toy examples
3. Abstention metric math (Precision, Recall, F1, over-abstention) on toy examples
4. Exclusion filter correctness (guarantees excluded docs never appear in retrieved_docs)
5. Bootstrap confidence intervals and paired test math sanity
"""

import pytest
import numpy as np

from eval.common import (
    normalize_answer,
    exact_match_score,
    token_f1_score,
    compute_retrieval_metrics,
    is_abstained_answer,
    UNANSWERABLE_STRING,
)
from eval.stats import compute_bootstrap_ci, paired_bootstrap_test


def test_answer_normalization():
    # Punctuation removal
    assert normalize_answer("Hello, World!") == "hello world"
    # Article removal
    assert normalize_answer("The quick brown fox and a lazy dog") == "quick brown fox and lazy dog"
    # Whitespace normalization
    assert normalize_answer("   Multiple   spaces \n\t tabs ") == "multiple spaces tabs"
    # HotpotQA official examples
    assert normalize_answer("Arthur's Magazine") == "arthurs magazine"
    assert normalize_answer("Delhi, India.") == "delhi india"


def test_exact_match_and_token_f1():
    # Exact match
    assert exact_match_score("Arthur's Magazine", "arthur's magazine") is True
    assert exact_match_score("The Arthur's Magazine", "Arthur's Magazine") is True
    assert exact_match_score("Harper's Magazine", "Arthur's Magazine") is False

    # Token F1
    f1, prec, rec = token_f1_score("President Richard Nixon", "Richard Nixon")
    # Pred: ["president", "richard", "nixon"] (3)
    # Gold: ["richard", "nixon"] (2)
    # Overlap: 2 -> prec = 2/3, rec = 2/2 = 1.0, f1 = 2 * (2/3) * 1 / (5/3) = 4/5 = 0.8
    assert round(prec, 4) == round(2 / 3, 4)
    assert round(rec, 4) == 1.0
    assert round(f1, 4) == 0.8

    # Completely disjoint
    f1_0, prec_0, rec_0 = token_f1_score("Paris, France", "Tokyo, Japan")
    assert f1_0 == 0.0 and prec_0 == 0.0 and rec_0 == 0.0

    # Both empty
    f1_empty, _, _ = token_f1_score("", "")
    assert f1_empty == 1.0


def test_retrieval_metrics_toy():
    gold = {"Doc A", "Doc B"}
    # Scenario 1: Retreived [Doc A, Doc C, Doc B, Doc D]
    retrieved = [
        {"title": "Doc A", "doc_id": "1"},
        {"title": "Doc C", "doc_id": "2"},
        {"title": "Doc B", "doc_id": "3"},
        {"title": "Doc D", "doc_id": "4"},
    ]
    m = compute_retrieval_metrics(retrieved, gold, k_list=(2, 3, 5))
    # At k=2: sub_docs = [Doc A, Doc C]. Hits = 1 ("Doc A").
    assert m["recall@2"] == 0.5
    assert m["precision@2"] == 0.5

    # At k=3: sub_docs = [Doc A, Doc C, Doc B]. Hits = 2.
    assert m["recall@3"] == 1.0
    assert round(m["precision@3"], 4) == round(2 / 3, 4)

    # Multi-hop success (both retrieved in top context)
    assert m["both_gold_retrieved"] == 1.0

    # Scenario 2: Only 1 gold retrieved
    retrieved_partial = [
        {"title": "Doc A", "doc_id": "1"},
        {"title": "Doc X", "doc_id": "5"},
    ]
    m_part = compute_retrieval_metrics(retrieved_partial, gold, k_list=(2,))
    assert m_part["recall@2"] == 0.5
    assert m_part["precision@2"] == 0.5
    assert m_part["both_gold_retrieved"] == 0.0


def test_abstention_math():
    # UNANSWERABLE set (positive class: should abstain)
    # 10 UNANSWERABLE questions: 8 abstained (TP=8, FN=2)
    # 10 ANSWERABLE questions: 1 abstained (FP=1, TN=9)
    tp = 8
    fn = 2
    fp = 1
    tn = 9

    precision = tp / (tp + fp)  # 8 / 9 = 0.8889
    recall = tp / (tp + fn)     # 8 / 10 = 0.8
    f1 = (2 * precision * recall) / (precision + recall)
    over_abstention = fp / (fp + tn)  # 1 / 10 = 0.1

    assert round(precision, 4) == 0.8889
    assert round(recall, 4) == 0.8
    assert round(f1, 4) == round((2 * (8/9) * 0.8) / ((8/9) + 0.8), 4)
    assert round(over_abstention, 4) == 0.1

    # Check is_abstained_answer function
    assert is_abstained_answer(UNANSWERABLE_STRING) is True
    assert is_abstained_answer("The relevant document is not present in the corpus.") is True
    assert is_abstained_answer("the relevant document is not present in the corpus") is True
    assert is_abstained_answer("The relevant document is not present in the corpus. Therefore I cannot answer.") is True
    assert is_abstained_answer("Arthur's Magazine was founded in 1844.") is False

    from eval.common import is_strict_abstained, is_soft_abstained, extract_answer_text
    # Strict vs Soft abstention
    assert is_strict_abstained("The relevant document is not present in the corpus.") is True
    assert is_strict_abstained("The documents do not state who the founder was.") is False
    assert is_soft_abstained("The documents do not state who the founder was.") is True
    assert is_soft_abstained("The evidence does not establish the year.") is True
    assert is_soft_abstained("This cannot be determined from the documents.") is True
    assert is_soft_abstained("He was born in London.") is False

    # extract_answer_text
    output_sample = "ANSWER: Richard Nixon\nSOURCES: [1, 2]"
    assert extract_answer_text(output_sample) == "Richard Nixon"
    output_sample_2 = "ANSWER: yes"
    assert extract_answer_text(output_sample_2) == "yes"
    output_sample_abs = "The relevant document is not present in the corpus."
    assert extract_answer_text(output_sample_abs) == UNANSWERABLE_STRING


def test_exclusion_filter_logic():
    from eval.caro_core import CorpusAssets, search_dense, search_sparse, search_hybrid, ExclusionContext

    CorpusAssets.load()
    query = "Radio City Indian radio station"

    # Get baseline dense results
    baseline_results = search_dense(query, top_k=5)
    assert len(baseline_results) == 5
    first_doc_id = baseline_results[0]["doc_id"]
    second_doc_id = baseline_results[1]["doc_id"]
    excluded_ids = {first_doc_id, second_doc_id}

    # Search with exclusion filter
    with ExclusionContext(excluded_ids):
        filtered_dense = search_dense(query, top_k=5)
        filtered_sparse = search_sparse(query, top_k=5)
        filtered_hybrid = search_hybrid(query, top_k=5)

    # Assert excluded docs never appear in ANY retriever
    filtered_dense_ids = {d["doc_id"] for d in filtered_dense}
    filtered_sparse_ids = {d["doc_id"] for d in filtered_sparse}
    filtered_hybrid_ids = {d["doc_id"] for d in filtered_hybrid}

    assert not (excluded_ids & filtered_dense_ids), f"Leaked docs in dense: {excluded_ids & filtered_dense_ids}"
    assert not (excluded_ids & filtered_sparse_ids), f"Leaked docs in sparse: {excluded_ids & filtered_sparse_ids}"
    assert not (excluded_ids & filtered_hybrid_ids), f"Leaked docs in hybrid: {excluded_ids & filtered_hybrid_ids}"

    # Assert that top_k is still full (5 docs returned)
    assert len(filtered_dense) == 5
    assert len(filtered_sparse) == 5
    assert len(filtered_hybrid) == 5


def test_bootstrap_ci_sanity():
    # Constant array: CI lower == upper == mean
    const_vals = [1.0] * 50
    ci_const = compute_bootstrap_ci(const_vals, n_resamples=500, seed=42)
    assert ci_const["mean"] == 1.0
    assert ci_const["ci_lower"] == 1.0
    assert ci_const["ci_upper"] == 1.0

    # Normal distribution with mean 10.0
    rng = np.random.RandomState(42)
    sample = list(rng.normal(loc=10.0, scale=1.0, size=100))
    ci_res = compute_bootstrap_ci(sample, n_resamples=1000, seed=42)
    assert 9.5 < ci_res["mean"] < 10.5
    assert ci_res["ci_lower"] < ci_res["mean"] < ci_res["ci_upper"]
    assert ci_res["ci_lower"] > 9.0 and ci_res["ci_upper"] < 11.0

    # Paired test sanity: clear superiority
    caro = [0.9] * 30
    baseline = [0.3] * 30
    pt = paired_bootstrap_test(caro, baseline, n_resamples=500, seed=42)
    assert pt["verdict"] == "CARO significantly better"
    assert pt["diff_mean"] == 0.6
    assert pt["diff_ci_lower"] == 0.6
    assert pt["p_value"] == 0.0

    # Paired test sanity: no difference
    caro_same = [0.5] * 30
    baseline_same = [0.5] * 30
    pt_same = paired_bootstrap_test(caro_same, baseline_same, n_resamples=500, seed=42)
    assert pt_same["verdict"] == "No significant difference"
    assert pt_same["diff_mean"] == 0.0


def test_seed123_sampling_and_disjointness():
    """Verifies that seed-123 sampling produces questions disjoint from seed-42."""
    import json
    from eval.caro_core import PROJECT_ROOT
    from eval.dataset import QUESTIONS_FILE, get_questions_filepath

    file42 = QUESTIONS_FILE
    file123 = get_questions_filepath(123)

    assert file42.exists(), f"Seed 42 questions file missing: {file42}"
    assert file123.exists(), f"Seed 123 questions file missing: {file123}"

    with open(file42, "r", encoding="utf-8") as f:
        q42_ids = {json.loads(line)["question_id"] for line in f if line.strip()}

    with open(file123, "r", encoding="utf-8") as f:
        q123_ids = {json.loads(line)["question_id"] for line in f if line.strip()}

    assert len(q42_ids) > 0, "Seed 42 questions should not be empty"
    assert len(q123_ids) > 0, "Seed 123 questions should not be empty"
    overlap = q42_ids & q123_ids
    assert len(overlap) == 0, f"Seed 123 must exclude all seed 42 question IDs! Found overlap: {overlap}"


def test_run_config_records_workers_and_device(tmp_path, monkeypatch):
    """Verifies that save_run_config records workers and device in run_config.json."""
    import json
    from eval.runner import save_run_config
    import eval.runner as runner_mod

    # Redirect RESULTS_DIR to tmp_path
    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)

    config_path = save_run_config(
        seed=123,
        sample_size=30,
        smoke=False,
        systems=["B1_Dense", "CARO"],
        workers=4,
        device="cuda"
    )

    assert config_path.exists()
    with open(config_path, "r", encoding="utf-8") as f:
        saved_config = json.load(f)

    assert saved_config["workers"] == 4
    assert saved_config["device"] == "cuda"
    assert saved_config["seed"] == 123
    assert saved_config["sample_size"] == 30

