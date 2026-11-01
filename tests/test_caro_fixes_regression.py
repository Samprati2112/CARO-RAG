from __future__ import annotations

import pytest

from modules.caro_pipeline import CAROPipeline
from modules.module3_adaptive_query_router.router import AdaptiveQueryRouter
from modules.module6_temporary_memory.memory_manager import TemporaryMemoryManager
from modules.module7_evidence_sufficiency.evidence_sufficiency import EvidenceSufficiencyAnalyzer
from modules.module8_answer_generation.generator import GroundedAnswerGenerator
from modules.module8_answer_generation.validator import AnswerValidator


def test_problem_a_direct_general_questions():
    """Verify Problem A: DIRECT route generates parametric answers without empty-context traps."""
    generator = GroundedAnswerGenerator(model_name="dummy-model")

    # Photosynthesis
    res1 = generator.generate("What is photosynthesis?", [], route="DIRECT")
    assert "photosynthesis" in res1.final_answer.lower()
    assert res1.validation.status == "PARAMETRIC_VALIDATED"
    assert res1.is_grounded is False

    # Artificial Intelligence
    res2 = generator.generate("What is artificial intelligence?", [], route="DIRECT")
    assert "artificial intelligence" in res2.final_answer.lower()
    assert res2.validation.status == "PARAMETRIC_VALIDATED"
    assert res2.is_grounded is False

    # Comparison / definitional query
    router = AdaptiveQueryRouter()
    routing = router.route("What is photosynthesis?")
    assert routing.route == "DIRECT"
    assert routing.question_type == "definitional"


def test_problem_b_sufficiency_distinguishes_lexical_overlap_from_factual_support():
    """Verify Problem B: M7 does not mark evidence sufficient solely due to keyword overlap."""
    analyzer = EvidenceSufficiencyAnalyzer()

    # Query: What is the capital of France?
    # Misleading distractor documents with keyword overlap ("capital", "France", "government")
    distractors = [
        {"doc_id": "d1", "text": "Luxembourg City is the capital of Luxembourg.", "normalized_score": 0.8},
        {"doc_id": "d2", "text": "Catherine Deneuve was born in France.", "normalized_score": 0.75},
        {"doc_id": "d3", "text": "The government of Berlin met in 1990.", "normalized_score": 0.6},
    ]
    res_distractor = analyzer.analyze("What is the capital of France?", distractors)
    assert res_distractor.sufficient is False
    assert res_distractor.action in {"RETRIEVE_AGAIN", "ABSTAIN"}

    # Relevant document actually stating the fact
    relevant = [
        {"doc_id": "d4", "text": "Paris is the capital of France.", "normalized_score": 0.95},
    ]
    res_relevant = analyzer.analyze("What is the capital of France?", relevant)
    assert res_relevant.sufficient is True
    assert res_relevant.action == "ANSWER"


def test_problem_b_world_war_and_government_branches_sufficiency():
    """Verify factual support gating on WWII and US government branches."""
    analyzer = EvidenceSufficiencyAnalyzer()

    # WWII: Mentioning the war without end date
    ww2_distractor = [
        {"doc_id": "w1", "text": "World War II was a global war involving over 30 countries.", "normalized_score": 0.8},
    ]
    assert analyzer.analyze("When did World War II end?", ww2_distractor).sufficient is False

    # WWII: Stating end date
    ww2_relevant = [
        {"doc_id": "w2", "text": "World War II ended on September 2, 1945 with the surrender of Japan.", "normalized_score": 0.95},
    ]
    assert analyzer.analyze("When did World War II end?", ww2_relevant).sufficient is True

    # US Government: Mentioning government branches generally
    gov_distractor = [
        {"doc_id": "g1", "text": "The US government operates branches across federal agencies.", "normalized_score": 0.7},
    ]
    assert analyzer.analyze("What are the three branches of the US government?", gov_distractor).sufficient is False

    # US Government: Naming the branches
    gov_relevant = [
        {"doc_id": "g2", "text": "The three branches of the US government are the executive, legislative, and judicial branches.", "normalized_score": 0.95},
    ]
    assert analyzer.analyze("What are the three branches of the US government?", gov_relevant).sufficient is True


def test_problem_c_memory_routing_standalone_vs_followup():
    """Verify Problem C: Standalone queries route to RETRIEVE, while legitimate follow-ups route to MEMORY."""
    router = AdaptiveQueryRouter()

    # Standalone queries that previously false-positively routed to MEMORY
    assert router.route("List the countries that border Germany.").route == "RETRIEVE"
    assert router.route("Which country hosted the Summer Olympics in 2016, and what is its capital?").route == "RETRIEVE"
    assert router.route("What evidence supports the claim that smoking increases the risk of lung cancer?").route == "RETRIEVE"
    assert router.route("What is the exact temperature inside my refrigerator right now?").route != "MEMORY"

    # Legitimate conversational follow-up queries
    r_follow1 = router.route("What about its advantages?")
    assert r_follow1.route == "MEMORY"
    assert r_follow1.requires_memory is True

    r_follow2 = router.route("Who was the person mentioned earlier?")
    assert r_follow2.route == "MEMORY"
    assert r_follow2.requires_memory is True

    r_follow3 = router.route("What were his most important scientific contributions?")
    assert r_follow3.requires_memory is True

    r_follow4 = router.route("Which of those contributions led to his Nobel Prize?")
    assert r_follow4.requires_memory is True


def test_problem_c_standalone_refrigerator_abstains_safely():
    """Verify that private real-time device questions safely abstain rather than inventing access."""
    generator = GroundedAnswerGenerator(model_name="dummy-model")
    res = generator.generate(
        "What is the exact temperature inside my refrigerator right now?",
        [],
        route="DIRECT",
    )
    assert res.validation.status == "ABSTAINED"
    assert "insufficient" in res.final_answer.lower()

    # If an answer falsely hallucinated a temperature, validator rejects it
    validator = AnswerValidator()
    hallucinated = validator.validate(
        "The temperature inside your refrigerator is 38 degrees Fahrenheit.",
        [],
        query="What is the exact temperature inside my refrigerator right now?",
        route="DIRECT",
    )
    assert hallucinated.is_valid is False
    assert hallucinated.status == "UNGROUNDED"


def test_problem_d_validation_rejects_unsupported_and_divergent_claims():
    """Verify Problem D: Validator checks actual facts and rejects unsupported or ambiguous assertions."""
    validator = AnswerValidator()

    # 1. Fully supported answer
    evidence = [{"text": "Jane Austen wrote Pride and Prejudice in 1813."}]
    good_res = validator.validate("Jane Austen wrote Pride and Prejudice.", evidence)
    assert good_res.is_valid is True
    assert good_res.is_grounded is True
    assert good_res.status == "VALIDATED"

    # 2. Unsupported factual claim
    bad_res = validator.validate(
        "Jane Austen wrote Pride and Prejudice in Tokyo with Charles Dickens.",
        evidence,
    )
    assert bad_res.is_valid is False
    assert bad_res.status == "UNGROUNDED"
    assert any("Tokyo" in e or "Charles Dickens" in e for e in bad_res.unsupported_entities)

    # 3. Ambiguous question over-specification
    ambiguous_res = validator.validate(
        "Barack Hussein Obama II was the first African American to have served as president.",
        [{"text": "Barack Obama was the first African American president."}],
        query="Who was the first president?",
    )
    assert ambiguous_res.is_valid is False
    assert ambiguous_res.status == "UNGROUNDED"

    # 4. Question with insufficient evidence (honest abstention is validated)
    abstain_res = validator.validate(
        "The provided evidence is insufficient to answer the query.",
        evidence,
        query="Who was the first emperor of China?",
    )
    assert abstain_res.is_valid is True
    assert abstain_res.status == "ABSTAINED"


def test_problem_d_conversational_coreference_consistency():
    """Verify that follow-up answers must match the conversational subject from memory."""
    validator = AnswerValidator()

    # Memory context discusses Albert Einstein
    evidence_with_memory = {
        "external_evidence": [{"text": "Subrahmanyan Chandrasekhar won the Nobel Prize in 1983."}],
        "memory_context": [{"user_query": "Tell me about Albert Einstein.", "stored_response": "Albert Einstein was a physicist."}],
    }

    # Answer claims Chandrasekhar in response to Einstein follow-up
    res = validator.validate(
        "Subrahmanyan Chandrasekhar was awarded the Nobel Prize for his stellar studies.",
        evidence_with_memory,
        query="Which of those contributions led to his Nobel Prize?",
    )
    assert res.is_valid is False
    assert res.status == "UNGROUNDED"
    assert "diverges from the conversational entity" in res.reasons[0]


def test_germany_false_border_validation_rejected():
    """Verify that claims substituting distractor geography for Germany are rejected as UNGROUNDED."""
    validator = AnswerValidator()
    evidence = [
        {"text": "The government of Berlin is the state government of Berlin, Germany."},
        {"text": "India is bounded by the Indian Ocean, the Arabian Sea, and the Bay of Bengal. It shares land borders with Pakistan, China, Nepal, and Bhutan."},
    ]
    query = "List the countries that border Germany."
    answer = "Germany is bounded by the Indian Ocean, the Arabian Sea, and the Bay of Bengal. It shares land borders with Pakistan, China, Nepal, and Bhutan."

    res = validator.validate(answer, evidence, query=query)
    assert res.is_valid is False
    assert res.status == "UNGROUNDED"
    assert res.grounding_score < 0.20
    assert any("Pakistan" in e for e in res.unsupported_entities)
    assert any("Indian Ocean" in e for e in res.unsupported_entities)


def test_theory_of_relativity_chandrasekhar_attribution_rejected():
    """Verify that false attribution of relativity to Chandrasekhar is rejected as UNGROUNDED."""
    validator = AnswerValidator()
    evidence = [
        {"text": "The Chandrasekhar limit was named after Subrahmanyan Chandrasekhar, the Indian astrophysicist who discovered it in 1930 in India."},
    ]
    query = "Which scientist developed the theory of relativity, and in which country was he born?"
    answer = "Subrahmanyan Chandrasekhar, an Indian American astrophysicist, developed the theory of relativity. He was born in India."

    res = validator.validate(answer, evidence, query=query)
    assert res.is_valid is False
    assert res.status == "UNGROUNDED"
    assert "Subrahmanyan Chandrasekhar" in res.unsupported_entities


def test_intra_sentential_pronouns_not_routed_to_memory():
    """Verify that queries with intra-sentential bound pronouns route to RETRIEVE, not MEMORY."""
    from modules.module3_adaptive_query_router.router import AdaptiveQueryRouter
    router = AdaptiveQueryRouter()

    # Bound pronouns in compound / relative inquiries
    r_relativity = router.route("Which scientist developed the theory of relativity, and in which country was he born?")
    assert r_relativity.route == "RETRIEVE"
    assert r_relativity.requires_memory is False

    r_tides = router.route("Why do tides occur, and which celestial body has the greatest influence on them?")
    assert r_tides.route == "RETRIEVE"
    assert r_tides.requires_memory is False


def test_einstein_three_turn_conversational_sequence_pipeline(tmp_path):
    """Verify that a 3-turn conversational sequence maintains entity isolation and subject across all turns."""
    from modules.caro_pipeline import CAROPipeline
    from modules.module6_temporary_memory.memory_manager import TemporaryMemoryManager
    from modules.module8_answer_generation.generator import GroundedAnswerGenerator

    class MockConversationalGenerator(GroundedAnswerGenerator):
        def generate(self, query, evidence, ground_truth=None, route="RETRIEVE", sufficiency_result=None):
            evidence_texts = self._extract_evidence_texts(evidence)
            if route == "DIRECT" and "Albert Einstein" in query:
                ans = "Albert Einstein (1879-1955) was a German-born theoretical physicist who developed the theory of relativity and won the 1921 Nobel Prize for the photoelectric effect."
            elif route == "MEMORY":
                combined = " ".join(evidence_texts)
                assert "Albert Einstein" in combined, f"Conversational context must preserve Albert Einstein, got: {combined}"
                assert "Chandrasekhar" not in combined, "Unrelated scientist leaked into conversational context"
                if "contributions" in query.lower():
                    ans = "Albert Einstein's most important scientific contributions include the theory of relativity and the explanation of the photoelectric effect."
                elif "nobel prize" in query.lower():
                    ans = "Albert Einstein's work on the photoelectric effect led to his Nobel Prize in Physics."
                else:
                    ans = "Albert Einstein made revolutionary contributions."
            else:
                ans = "General response"

            validation = self.validator.validate(ans, evidence, query=query, route=route)
            return self._build_result(query, ans, validation, evidence_texts)

        def _build_result(self, query, ans, validation, texts):
            from modules.module8_answer_generation.schemas import AnswerGenerationResult
            return AnswerGenerationResult(
                query=query,
                raw_response=ans,
                final_answer=ans if validation.is_valid else f"Validation notice: {validation.reasons[0]}",
                is_grounded=validation.is_grounded,
                grounding_score=validation.grounding_score,
                validation=validation,
                evaluation=None,
                evidence_count=len(texts),
                model_name="mock",
            )

    memory_manager = TemporaryMemoryManager(storage_path=tmp_path)
    mock_gen = MockConversationalGenerator(model_name="dummy-model")
    pipeline = CAROPipeline(generator=mock_gen, memory_manager=memory_manager)
    pipeline.conversation_id = "test_conv_einstein"

    # Turn 1
    t1 = pipeline.answer_query("Tell me about Albert Einstein.", turn_id="1")
    assert t1.routing.route == "DIRECT"
    assert "Albert Einstein" in t1.answer
    assert t1.answer_result.validation.status == "PARAMETRIC_VALIDATED"

    # Turn 2
    t2 = pipeline.answer_query("What were his most important scientific contributions?", turn_id="2")
    assert t2.routing.route == "MEMORY"
    assert "Albert Einstein" in t2.answer
    assert "Chandrasekhar" not in t2.answer
    assert t2.answer_result.validation.is_valid is True

    # Turn 3
    t3 = pipeline.answer_query("Which of those contributions led to his Nobel Prize?", turn_id="3")
    assert t3.routing.route == "MEMORY"
    assert "photoelectric effect" in t3.answer.lower()
    assert "Albert Einstein" in t3.answer
    assert "Chandrasekhar" not in t3.answer
    assert t3.answer_result.validation.is_valid is True

