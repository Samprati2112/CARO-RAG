from __future__ import annotations

import pytest

from modules.caro_pipeline import CAROPipeline
from modules.module6_temporary_memory.memory_manager import TemporaryMemoryManager
from modules.module7_evidence_sufficiency.evidence_sufficiency import EvidenceSufficiencyAnalyzer
from modules.module8_answer_generation.generator import GroundedAnswerGenerator
from modules.module8_answer_generation.validator import AnswerValidator


# =====================================================================
# Test A: Frankenstein Evidence (Fix A)
# =====================================================================
def test_frankenstein_rejects_bride_of_frankenstein_and_actress():
    """Test A: Rejects Bride of Frankenstein movie stubs and Elsa Lanchester as spouse."""
    analyzer = EvidenceSufficiencyAnalyzer()
    validator = AnswerValidator()

    # Distractor evidence from 1935 movie stubs
    distractor_docs = [
        {
            "doc_id": "hotpotqa_5a864b81554299211dda2aea_4",
            "title": "Bride of Frankenstein",
            "text": "Bride of Frankenstein is a 1935 American science-fiction horror film directed by James Whale. The sequel features Elsa Lanchester in the dual role of Mary Shelley and the Monster's mate at the end of the film.",
        },
        {
            "doc_id": "hotpotqa_5a78d55555429970f5fffd9a_9",
            "title": "Monsters of Legend",
            "text": "Monsters of Legend features images from the 1935 film Bride of Frankenstein starring Boris Karloff and Elsa Lanchester.",
        },
    ]

    query = "Who was the spouse of the author who wrote Frankenstein?"

    # 1. Evidence Assessor must reject Bride of Frankenstein as evidence for the spouse relation
    suff_res = analyzer.analyze(query, distractor_docs)
    assert suff_res.sufficient is False
    assert suff_res.status in ("INSUFFICIENT", "PARTIAL")
    assert "spouse of Mary Shelley" in suff_res.missing_components or "spouse of Mary Shelley" in suff_res.unsupported_claims
    assert "actress" not in suff_res.reason.lower()

    # 2. Generator must not return the film's actress Elsa Lanchester as the spouse
    generator = GroundedAnswerGenerator(model_name="dummy-model")
    gen_res = generator.generate(query, distractor_docs, sufficiency_result=suff_res)
    assert "elsa lanchester" not in gen_res.final_answer.lower() or "spouse" not in gen_res.final_answer.lower()
    # It must safely abstain or reject rather than returning Elsa Lanchester as the spouse
    assert gen_res.citation_status in ("REJECTED", "ABSTAINED") or "not present in the corpus" in gen_res.final_answer.lower()

    # 3. If Mary Shelley document WITH Percy Bysshe Shelley is provided, it succeeds
    grounded_docs = [
        {
            "doc_id": "hotpotqa_5a8f4b0c5542997ba9cb3225_9",
            "title": "Mary Shelley",
            "text": "Mary Wollstonecraft Shelley was an English novelist, best known for her Gothic novel Frankenstein (1818). Her husband, the Romantic poet and philosopher Percy Bysshe Shelley, encouraged her writing.",
        }
    ]
    good_suff = analyzer.analyze(query, grounded_docs)
    assert good_suff.sufficient is True
    assert good_suff.status == "SUFFICIENT"
    assert "spouse of author" in good_suff.supporting_doc_ids

    good_gen = generator.generate(query, grounded_docs, sufficiency_result=good_suff)
    assert "percy bysshe shelley" in good_gen.final_answer.lower()
    assert "mary shelley" in good_gen.final_answer.lower()
    assert good_gen.validation.is_valid is True
    assert good_gen.citation_status == "VALIDATED"


# =====================================================================
# Test B: Tides and Citation Integrity (Fix B)
# =====================================================================
def test_tides_rejects_infrastructure_citations_and_abstains():
    """Test B: Rejects civil infrastructure documents as proof of astronomical tides."""
    analyzer = EvidenceSufficiencyAnalyzer()
    validator = AnswerValidator()
    generator = GroundedAnswerGenerator(model_name="dummy-model")

    query = "What causes ocean tides?"

    # Retrieved passages about Miami Beach storm-water flood pumps
    infrastructure_docs = [
        {
            "doc_id": "hotpotqa_5ae1cbfb554299234fd04300_8",
            "title": "Tidal flooding",
            "text": "Tidal flooding is the temporary inundation of low-lying areas, especially streets, during exceptionally high tides. In Miami Beach, Florida, storm pumps and raised roads were installed under Governor Rick Scott to prevent sea water backup.",
        }
    ]

    # 1. Evidence Assessor rejects infrastructure as cause of tides
    suff = analyzer.analyze(query, infrastructure_docs)
    assert suff.sufficient is False
    assert suff.status == "INSUFFICIENT"

    # 2. If a model generates astronomical physics citing the Miami Beach document, validator rejects it
    hallucinated_answer = (
        "Ocean tides occur due to the gravitational pull of the Moon and the Sun on Earth's oceans [Document 1]. "
        "The gravitational force causes ocean waters to bulge [Document 1]."
    )
    val = validator.validate(hallucinated_answer, infrastructure_docs, query=query)
    assert val.is_valid is False
    assert val.status in ("UNGROUNDED", "INVALID_CITATION")
    # Verify claim validation explicitly caught that Document 1 does not support gravitational pull
    assert any(not cv["is_supported"] for cv in val.claim_validations)

    # 3. Generator must safely reject the citation and deliver honest abstention
    gen_res = generator.generate(query, infrastructure_docs, sufficiency_result=suff)
    assert "not present in the corpus" in gen_res.final_answer.lower() or "insufficient" in gen_res.final_answer.lower()
    assert gen_res.citation_status in ("REJECTED", "ABSTAINED")


# =====================================================================
# Test C: Einstein Memory Sequence (Fix C)
# =====================================================================
def test_einstein_memory_sequence_resolves_photoelectric_effect(tmp_path):
    """Test C: Resolves 'those contributions' to photoelectric effect for Nobel Prize."""
    memory_manager = TemporaryMemoryManager(storage_path=tmp_path)
    generator = GroundedAnswerGenerator(model_name="dummy-model")
    pipeline = CAROPipeline(generator=generator, memory_manager=memory_manager)
    pipeline.conversation_id = "test_einstein_seq"

    # Turn 1: Tell me about Albert Einstein.
    t1 = pipeline.answer_query("Tell me about Albert Einstein.", turn_id="1")
    assert t1.routing.route == "DIRECT"
    assert "albert einstein" in t1.answer.lower()
    assert "photoelectric effect" in t1.answer.lower()

    # Turn 2: What were his most important scientific contributions?
    t2 = pipeline.answer_query("What were his most important scientific contributions?", turn_id="2")
    assert t2.routing.route == "MEMORY"
    assert "relativity" in t2.answer.lower()
    assert t2.answer_result.validation.is_valid is True

    # Turn 3: Which of those contributions led to his Nobel Prize?
    t3 = pipeline.answer_query("Which of those contributions led to his Nobel Prize?", turn_id="3")
    assert t3.routing.route == "MEMORY"
    # MUST resolve photoelectric effect, NOT repeat relativity alone!
    assert "photoelectric effect" in t3.answer.lower()
    assert t3.answer_result.validation.is_valid is True
    # Provenance must link to earlier conversation context
    assert t3.stored_memory is not None


# =====================================================================
# Test D: Multi-Document Evidence (Fix A & D)
# =====================================================================
def test_multi_document_evidence_combination_and_partial_distinction():
    """Test D: Combines evidence across documents and distinguishes partial vs complete."""
    analyzer = EvidenceSufficiencyAnalyzer()

    # 1. Germany borders (Q18): 3 separate documents covering different borders
    germany_docs = [
        {"doc_id": "d_dk", "title": "Denmark", "text": "Denmark shares a land border with Germany to the south."},
        {"doc_id": "d_pl", "title": "Poland", "text": "Poland is bordered by Germany to the west and the Czech Republic to the south."},
        {"doc_id": "d_at", "title": "Austria", "text": "Austria shares borders with Germany and the Czech Republic to the north."},
    ]
    res_germany = analyzer.analyze("List the countries that border Germany.", germany_docs)
    # Assessor successfully combines across docs, but flags as PARTIAL since not all 9 are covered
    assert res_germany.status == "PARTIAL"
    assert res_germany.evidence_count == 3
    assert len(res_germany.supporting_doc_ids) > 0
    assert "Denmark" in str(res_germany.supporting_snippets)
    assert "Poland" in str(res_germany.supporting_snippets)
    assert "Austria" in str(res_germany.supporting_snippets)

    # 2. Solar system planets (Q20): 3 separate documents naming 5+ planets
    planet_docs = [
        {"doc_id": "p1", "title": "Outer Planets", "text": "The outer planets in our solar system are Jupiter, Saturn, Uranus, and Neptune."},
        {"doc_id": "p2", "title": "Earth", "text": "Earth is the third planet in the solar system from the Sun."},
        {"doc_id": "p3", "title": "Mars", "text": "Mars is the fourth planet from the Sun in the solar system."},
    ]
    res_planets = analyzer.analyze("Name five planets in our solar system.", planet_docs)
    # Multi-document combination yields 6 distinct planets >= 5 -> SUFFICIENT!
    assert res_planets.status == "SUFFICIENT"
    assert res_planets.sufficient is True
    assert res_planets.action == "ANSWER"


# =====================================================================
# Test E: Irrelevant but Topically Similar Evidence (Fix A & B)
# =====================================================================
def test_topical_distractors_rejected_by_assessor_and_validator():
    """Test E: Rejects topical keyword overlaps that lack the required factual predicate."""
    analyzer = EvidenceSufficiencyAnalyzer()
    validator = AnswerValidator()

    query = "Who was the spouse of the author who wrote Frankenstein?"
    topical_distractors = [
        {"doc_id": "t1", "text": "The Castle of Frankenstein was a Gothic structure that inspired horror stories."},
        {"doc_id": "t2", "text": "Frankenstein's monster had bolts on his neck in the stage adaptation."},
    ]

    suff = analyzer.analyze(query, topical_distractors)
    assert suff.sufficient is False
    assert suff.status == "INSUFFICIENT"

    # Unsupported answer claiming Castle architect as spouse
    bad_answer = "Frankenstein's author married Baron von Frankenstein [Document 1]."
    val = validator.validate(bad_answer, topical_distractors, query=query)
    assert val.is_valid is False
    assert val.status in ("UNGROUNDED", "INVALID_CITATION")


# =====================================================================
# Test F: Unsupported Citation (Fix B)
# =====================================================================
def test_unsupported_citation_detected_and_rejected():
    """Test F: Valid document ID whose text does not support the claim is rejected."""
    validator = AnswerValidator()

    valid_docs = [
        {"doc_id": "doc_1", "text": "The Walt Disney Company France is headquartered in Paris, France."},
        {"doc_id": "doc_2", "text": "Air France operates flights between Paris and Tokyo."},
    ]

    # Citing Document 1 for a completely unsupported claim
    fabricated_claim = "The French Revolution began when Napoleon marched into Berlin in 1789 [Document 1]."
    val = validator.validate(fabricated_claim, valid_docs, query="When did the French Revolution begin?")
    assert val.is_valid is False
    assert val.status in ("UNGROUNDED", "INVALID_CITATION")
    assert val.citation_accuracy_score < 0.5
    assert val.claim_validations[0]["citations_valid"] is False


# =====================================================================
# Test G: Follow-up with Recency Conflict (Fix C)
# =====================================================================
def test_followup_recency_conflict_resolves_correct_earlier_fact(tmp_path):
    """Test G: Resolves specific earlier fact rather than echoing conflicting recent turn."""
    store = TemporaryMemoryManager(storage_path=tmp_path).store

    # Turn 1 established Nobel Prize for Photoelectric Effect
    store.add(
        conversation_id="conv_recency_test",
        turn_id="1",
        user_query="Tell me about Einstein.",
        assistant_response="Einstein developed relativity and won the Nobel Prize for his explanation of the photoelectric effect.",
    )

    # Turn 2 discussed general contributions with relativity mentioned prominently
    store.add(
        conversation_id="conv_recency_test",
        turn_id="2",
        user_query="What was his most famous theory?",
        assistant_response="His most famous theory was the general theory of relativity.",
    )

    # Follow-up: "Which of those contributions led to his Nobel Prize?"
    matching_facts = store.search_facts("Which of those contributions led to his Nobel Prize?", "conv_recency_test")
    assert len(matching_facts) > 0
    # Top matching fact MUST be nobel_prize_reason (photoelectric effect), not relativity
    assert matching_facts[0].relation == "nobel_prize_reason"
    assert "photoelectric" in matching_facts[0].value.lower()
    assert matching_facts[0].source_turn == "1"


# =====================================================================
# Test H: Existing Behavior Regression
# =====================================================================
def test_existing_behavior_regression():
    """Test H: Ordinary grounded questions, valid abstentions, and responses still work."""
    generator = GroundedAnswerGenerator(model_name="dummy-model")
    validator = AnswerValidator()

    # 1. Direct grounded factual question
    france_docs = [{"text": "Paris is the capital of France.", "document_id": "d_paris"}]
    res_france = generator.generate("What is the capital of France?", france_docs)
    assert "paris" in res_france.final_answer.lower()
    assert res_france.validation.is_valid is True

    # 2. Honest abstention for empty evidence
    res_empty = generator.generate("What is the capital of Atlantis?", [])
    assert "insufficient" in res_empty.final_answer.lower() or "not present in the corpus" in res_empty.final_answer.lower()
    assert res_empty.validation.status == "ABSTAINED"

    # 3. Honest abstention for private data
    res_private = generator.generate("What did I eat for breakfast?", [], route="DIRECT")
    assert res_private.validation.status == "ABSTAINED"
