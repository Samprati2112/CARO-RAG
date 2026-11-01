from __future__ import annotations

import pytest

from modules.module8_answer_generation.evaluator import GenerationEvaluator
from modules.module8_answer_generation.generator import GroundedAnswerGenerator
from modules.module8_answer_generation.schemas import AnswerGenerationResult, EvaluationMetrics, ValidationResult
from modules.module8_answer_generation.validator import AnswerValidator


def test_generation_receives_query_and_validated_evidence():
    generator = GroundedAnswerGenerator(model_name="dummy-model")
    evidence = [{"text": "Paris is the capital city of France."}]
    result = generator.generate("What is the capital of France?", evidence)

    assert result.query == "What is the capital of France?"
    assert result.evidence_count == 1
    assert "Paris" in result.final_answer
    assert result.is_grounded is True
    assert result.validation.status == "VALIDATED"


def test_generation_is_grounded_in_supplied_context():
    validator = AnswerValidator()
    evidence = [{"text": "Acme Corp was founded by John Smith in 1995."}]
    
    # Grounded answer
    good_result = validator.validate("John Smith founded Acme Corp in 1995.", evidence)
    assert good_result.is_grounded is True
    assert good_result.status == "VALIDATED"
    assert good_result.grounding_score > 0.6

    # Ungrounded answer claiming fabricated entity
    bad_result = validator.validate("Acme Corp was founded by Alice Wonderland in Tokyo in 2020.", evidence)
    assert bad_result.is_grounded is False
    assert bad_result.status == "UNGROUNDED"
    assert any("Alice" in ent or "Tokyo" in ent or "2020" in ent for ent in bad_result.unsupported_entities)


def test_empty_evidence_is_handled_safely():
    generator = GroundedAnswerGenerator(model_name="dummy-model")
    result = generator.generate("What is the capital of France?", [])
    assert result.evidence_count == 0
    assert result.validation.status == "ABSTAINED"
    assert "insufficient" in result.final_answer.casefold()


def test_insufficient_evidence_is_not_silently_treated_as_sufficient():
    validator = AnswerValidator()
    # Explicit abstention response is recognized and validated as honest abstention
    res = validator.validate("The provided evidence is insufficient to answer the query.", [{"text": "Unrelated passage."}])
    assert res.status == "ABSTAINED"
    assert res.is_valid is True
    assert res.is_grounded is True


def test_malformed_and_empty_output_handling():
    validator = AnswerValidator()
    
    empty_res = validator.validate("", [{"text": "Some text."}])
    assert empty_res.status == "EMPTY"
    assert empty_res.is_valid is False

    malformed_res = validator.validate("Generation failed: Connection refused to Ollama.", [{"text": "Some text."}])
    assert malformed_res.status == "MALFORMED"
    assert malformed_res.is_valid is False


def test_deterministic_validation_behavior():
    validator = AnswerValidator()
    answer = "Paris is the capital of France."
    evidence = [{"text": "Paris is the capital of France."}]
    res1 = validator.validate(answer, evidence)
    res2 = validator.validate(answer, evidence)
    assert res1.to_dict() == res2.to_dict()


def test_evaluation_metrics_exact_match_and_f1():
    evaluator = GenerationEvaluator()
    
    # Exact Match
    assert evaluator.compute_exact_match("Paris", "paris") == 1.0
    assert evaluator.compute_exact_match("Paris.", "The Paris") == 1.0
    assert evaluator.compute_exact_match("London", "Paris") == 0.0

    # Token F1
    f1_perfect = evaluator.compute_f1("Paris is capital", "Paris is the capital")
    assert f1_perfect == 1.0

    f1_partial = evaluator.compute_f1("Paris is capital of France", "Paris is in France")
    assert 0.0 < f1_partial < 1.0

    f1_zero = evaluator.compute_f1("London UK", "Paris France")
    assert f1_zero == 0.0


def test_evaluation_metrics_grounding_ratio():
    evaluator = GenerationEvaluator()
    evidence_texts = ["Paris is the capital of France, located in Europe."]
    
    # Fully grounded prediction
    ratio_high = evaluator.compute_grounding_ratio("Paris is capital of France", evidence_texts)
    assert ratio_high == 1.0

    # Prediction with hallucinated terms
    ratio_mixed = evaluator.compute_grounding_ratio("Paris is located in Australia near Sydney", evidence_texts)
    assert ratio_mixed < 0.6


def test_no_retrieval_performed_inside_m8():
    generator = GroundedAnswerGenerator(model_name="dummy-model")
    # Verify M8 generator has no retriever attributes or network retrieval methods
    assert not hasattr(generator, "retriever")
    assert not hasattr(generator, "vector_store")
    assert not hasattr(generator, "bm25")
    assert not hasattr(generator, "faiss")


def test_generator_with_ground_truth_computes_evaluation():
    generator = GroundedAnswerGenerator(model_name="dummy-model")
    evidence = [{"text": "Paris is the capital of France."}]
    result = generator.generate(
        "What is the capital of France?",
        evidence,
        ground_truth="Paris is the capital of France.",
    )
    assert result.evaluation is not None
    assert result.evaluation.exact_match == 1.0
    assert result.evaluation.f1_score == 1.0
    assert result.evaluation.grounding_ratio == 1.0
