from copy import deepcopy

from modules.module7_evidence_sufficiency import EvidenceSufficiencyAnalyzer, SufficiencyConfig


def evidence(text, score=0.9, selected=False):
    return {
        "doc_id": text[:12],
        "text": text,
        "normalized_score": score,
        "selected_for_evidence": selected,
        "metadata": {"source": "test"},
    }


def test_empty_evidence_is_insufficient():
    result = EvidenceSufficiencyAnalyzer().analyze("What is the capital of France?", [])
    assert result.sufficient is False
    assert result.action == "RETRIEVE_AGAIN"
    assert result.evidence_count == 0


def test_relevant_single_hop_evidence_is_sufficient():
    result = EvidenceSufficiencyAnalyzer().analyze(
        "What is the capital of France?",
        [evidence("Paris is the capital of France.")],
    )
    assert result.sufficient is True
    assert result.action == "ANSWER"
    assert result.coverage_ratio >= 0.75


def test_irrelevant_evidence_is_insufficient():
    result = EvidenceSufficiencyAnalyzer().analyze(
        "What is the capital of France?",
        [evidence("Berlin is the capital of Germany.", score=0.1)],
    )
    assert result.sufficient is False
    assert result.missing_aspects == ["query terms: france"]


def test_incomplete_multi_hop_evidence_is_insufficient():
    result = EvidenceSufficiencyAnalyzer().analyze(
        "Which country was the birthplace of the person who founded the company?",
        [
            evidence("John Smith founded Acme Company.", score=0.9),
            evidence("John Smith was born in Paris.", score=0.9),
        ],
    )
    assert result.sufficient is False
    assert "country or location" in result.missing_aspects


def test_complete_multi_hop_evidence_is_sufficient():
    result = EvidenceSufficiencyAnalyzer().analyze(
        "Which country was the birthplace of the person who founded the company?",
        [
            evidence("John Smith founded Acme Company.", score=0.9),
            evidence("John Smith was born in Paris.", score=0.9),
            evidence("Paris is in France.", score=0.9),
        ],
    )
    assert result.sufficient is True
    assert result.action == "ANSWER"
    assert result.missing_aspects == []


def test_insufficient_evidence_retrieves_again_before_budget():
    result = EvidenceSufficiencyAnalyzer().analyze("What is the capital of France?", [], iteration=2, max_iterations=5)
    assert result.action == "RETRIEVE_AGAIN"


def test_insufficient_evidence_abstains_at_budget():
    result = EvidenceSufficiencyAnalyzer().analyze("What is the capital of France?", [], iteration=5, max_iterations=5)
    assert result.action == "ABSTAIN"
    assert "retrieval budget" in result.reason


def test_repeated_evaluation_is_deterministic():
    analyzer = EvidenceSufficiencyAnalyzer()
    items = [evidence("Paris is the capital of France.")]
    assert analyzer.analyze("What is the capital of France?", items) == analyzer.analyze("What is the capital of France?", items)


def test_scores_and_coverage_are_present_and_consistent():
    result = EvidenceSufficiencyAnalyzer().analyze("What is the capital of France?", [evidence("Paris is the capital of France.")])
    assert 0.0 <= result.evidence_score <= 1.0
    assert 0.0 <= result.coverage_ratio <= 1.0
    assert result.evidence_count == 1
    assert result.to_dict()["coverage_ratio"] == result.coverage_ratio


def test_evidence_is_not_mutated():
    items = [evidence("Paris is the capital of France.")]
    before = deepcopy(items)
    EvidenceSufficiencyAnalyzer().analyze("What is the capital of France?", items)
    assert items == before


def test_disconnected_entities_multi_hop_fails_chain():
    # Terms occur across documents, but entity connections are disconnected
    result = EvidenceSufficiencyAnalyzer().analyze(
        "Which country was the birthplace of the person who founded the company?",
        [
            evidence("John Smith founded Acme Company.", score=0.9),
            evidence("David was born in Berlin.", score=0.9),
            evidence("Tokyo is in Japan.", score=0.9),
        ],
    )
    assert result.sufficient is False
    assert result.action == "RETRIEVE_AGAIN"


def test_configurable_threshold_behavior():
    strict_analyzer = EvidenceSufficiencyAnalyzer(SufficiencyConfig(coverage_threshold=0.95))
    result = strict_analyzer.analyze(
        "What is the capital of France?",
        [evidence("Paris is the capital of France.", score=0.5)],
    )
    assert result.sufficient is False
    assert result.action == "RETRIEVE_AGAIN"


def test_suggested_subquery_on_missing_aspect():
    result = EvidenceSufficiencyAnalyzer().analyze(
        "Which country was the birthplace of the person who founded the company?",
        [
            evidence("John Smith founded Acme Company.", score=0.9),
            evidence("John Smith was born in Paris.", score=0.9),
        ],
    )
    assert result.suggested_subquery is not None