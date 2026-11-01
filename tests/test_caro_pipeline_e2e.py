from __future__ import annotations

import numpy as np
import pytest

from modules.caro_pipeline import CAROPipeline
from modules.module5_retrieval.retriever import RetrievalEngine
from modules.module5_retrieval.schemas import RetrievalConfig
from modules.module6_temporary_memory.memory_manager import TemporaryMemoryManager
from modules.module7_evidence_sufficiency.evidence_sufficiency import EvidenceSufficiencyAnalyzer
from modules.module8_answer_generation.generator import GroundedAnswerGenerator


class FakeEmbedder:
    def encode_query(self, text):
        vec = np.ones(1024, dtype=np.float32)
        return vec / np.linalg.norm(vec)


class FakeDenseRetriever:
    def __init__(self, responses: dict[str, list[dict]]):
        self.responses = responses

    def retrieve(self, query: str, top_k: int = 5):
        for key, docs in self.responses.items():
            if key.lower() in query.lower():
                return docs
        return [{"doc_id": "fallback", "text": "Generic passage", "retrieval_score": 0.5}]


def test_e2e_direct_pipeline_flow(tmp_path):
    generator = GroundedAnswerGenerator(model_name="dummy-model")
    pipeline = CAROPipeline(
        conversation_id="e2e_direct",
        memory_manager=TemporaryMemoryManager(tmp_path, embedder=FakeEmbedder()),
        generator=generator,
    )
    result = pipeline.answer_query("What is Python?")
    assert result.routing.route == "DIRECT"
    assert result.strategy.selected_strategy == "DIRECT"
    assert result.retrieval is None
    assert result.answer_result is not None
    assert result.stored_memory.turn_id == "1"


def test_e2e_factual_single_hop_with_m7_m8(tmp_path):
    fake_retriever = FakeDenseRetriever({
        "france": [{"doc_id": "doc1", "text": "Paris is the capital of France.", "retrieval_score": 0.95}],
    })
    retrieval_engine = RetrievalEngine(
        RetrievalConfig(retrieval_top_k=1, relevance_threshold=0.5),
        dense_retriever=fake_retriever,
    )
    generator = GroundedAnswerGenerator(model_name="dummy-model")
    pipeline = CAROPipeline(
        conversation_id="e2e_factual",
        memory_manager=TemporaryMemoryManager(tmp_path, embedder=FakeEmbedder()),
        retrieval_engine=retrieval_engine,
        generator=generator,
    )
    result = pipeline.answer_query("What is the capital of France?")
    assert result.routing.route == "RETRIEVE"
    assert result.sufficiency is not None
    assert result.sufficiency.sufficient is True
    assert result.sufficiency.action == "ANSWER"
    assert result.answer_result is not None
    assert result.answer_result.is_grounded is True
    assert "Paris" in result.answer


def test_e2e_multi_hop_iterative_sufficiency(tmp_path):
    fake_retriever = FakeDenseRetriever({
        "birthplace": [
            {"doc_id": "d1", "text": "John Smith founded Acme Company.", "retrieval_score": 0.9},
            {"doc_id": "d2", "text": "John Smith was born in Paris.", "retrieval_score": 0.9},
            {"doc_id": "d3", "text": "Paris is in France.", "retrieval_score": 0.9},
        ],
    })
    retrieval_engine = RetrievalEngine(
        RetrievalConfig(retrieval_top_k=3, relevance_threshold=0.5),
        dense_retriever=fake_retriever,
        sparse_retriever=fake_retriever,
    )
    generator = GroundedAnswerGenerator(model_name="dummy-model")
    pipeline = CAROPipeline(
        conversation_id="e2e_multihop",
        memory_manager=TemporaryMemoryManager(tmp_path, embedder=FakeEmbedder()),
        retrieval_engine=retrieval_engine,
        generator=generator,
    )
    result = pipeline.answer_query("Which country was the birthplace of the person who founded the company?")
    assert result.routing.question_type == "multi-hop"
    assert result.strategy.selected_strategy == "HYBRID"
    assert result.sufficiency is not None
    assert result.sufficiency.sufficient is True
    assert result.sufficiency.chain_verified is True
    assert result.answer_result is not None
    assert result.answer_result.is_grounded is True


def test_e2e_abstention_when_evidence_remains_insufficient(tmp_path):
    fake_retriever = FakeDenseRetriever({
        "france": [{"doc_id": "doc1", "text": "Unrelated history text.", "retrieval_score": 0.1}],
    })
    retrieval_engine = RetrievalEngine(
        RetrievalConfig(retrieval_top_k=1, relevance_threshold=0.5),
        dense_retriever=fake_retriever,
    )
    generator = GroundedAnswerGenerator(model_name="dummy-model")
    pipeline = CAROPipeline(
        conversation_id="e2e_insufficient",
        memory_manager=TemporaryMemoryManager(tmp_path, embedder=FakeEmbedder()),
        retrieval_engine=retrieval_engine,
        generator=generator,
        max_retrieval_iterations=2,
    )
    result = pipeline.answer_query("What is the capital of France?")
    assert result.sufficiency is not None
    assert result.sufficiency.sufficient is False
    assert result.sufficiency.action in {"RETRIEVE_AGAIN", "ABSTAIN"}
