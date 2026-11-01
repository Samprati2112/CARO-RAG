import pytest

from modules.caro_pipeline import CAROPipeline
from modules.module2_baseline_rag.generator import Generator
from modules.module5_retrieval.retriever import RetrievalEngine
from modules.module5_retrieval.schemas import RetrievalConfig
from modules.module6_temporary_memory.memory_manager import TemporaryMemoryManager


pytest.importorskip("sentence_transformers")


@pytest.fixture(scope="session")
def real_embedder():
    from modules.module2_baseline_rag.embeddings import EmbeddingModel

    try:
        return EmbeddingModel(device="cpu")
    except Exception as exc:
        pytest.skip(f"BGE-M3 is unavailable in this environment: {exc}")


class CapturingGenerator(Generator):
    def __init__(self):
        super().__init__(model_name="dummy-model")
        self.contexts = []

    def generate(self, query, context_documents):
        self.contexts.append(context_documents)
        if context_documents["memory_context"]:
            return "Python has advantages described by the previous turn."
        return "Python is a programming language."


class FakeDenseRetriever:
    def retrieve(self, query, top_k=None):
        return [{"doc_id": "external-1", "text": "External evidence", "retrieval_score": 1.0}]


def test_real_bge_embedding_persistence_reload_threshold_and_isolation(tmp_path, real_embedder):
    first_store = TemporaryMemoryManager(tmp_path, embedder=real_embedder, memory_relevance_threshold=0.2)
    record = first_store.add_memory(
        "conversation-a",
        "1",
        "What is Python?",
        "Python is a programming language.",
        "Python context",
    )
    assert len(record.embedding) == 1024
    assert abs(sum(value * value for value in record.embedding) ** 0.5 - 1.0) < 1e-5

    reloaded_store = TemporaryMemoryManager(tmp_path, embedder=real_embedder, memory_relevance_threshold=0.2)
    result = reloaded_store.retrieve_memory("What are its advantages?", "conversation-a")
    assert result.memory_found is True
    assert result.memories[0].turn_id == "1"
    assert result.memories[0].stored_response == "Python is a programming language."
    assert reloaded_store.retrieve_memory("What are its advantages?", "conversation-b").memory_found is False
    assert reloaded_store.retrieve_memory("What are its advantages?", "conversation-a", threshold=1.1).memory_found is False


def test_exact_two_turn_real_bge_end_to_end(tmp_path, real_embedder):
    memory = TemporaryMemoryManager(tmp_path, embedder=real_embedder, memory_relevance_threshold=0.2)
    generator = CapturingGenerator()
    pipeline = CAROPipeline(
        conversation_id="m6_e2e_test",
        memory_manager=memory,
        generator=generator,
    )

    first = pipeline.answer_query("What is Python?", turn_id="1")
    second = pipeline.answer_query("What are its advantages?", turn_id="2")

    assert first.routing.route == "DIRECT"
    assert first.stored_memory.turn_id == "1"
    assert second.routing.route == "MEMORY"
    assert second.strategy.selected_strategy == "MEMORY"
    assert second.memory is not None and second.memory.memory_found is True
    assert second.memory.memories[0].original_query == "What is Python?"
    assert second.memory.memories[0].stored_response == first.answer
    assert generator.contexts[1]["memory_context"]
    assert first.answer in generator.build_prompt("What are its advantages?", generator.contexts[1])
    assert "previous turn" in second.answer
    assert second.stored_memory.turn_id == "2"
    assert memory.get_memory_state("m6_e2e_test")["record_count"] == 2


def test_memory_and_retrieval_composes_real_memory_with_external_evidence(tmp_path, real_embedder):
    memory = TemporaryMemoryManager(tmp_path, embedder=real_embedder, memory_relevance_threshold=0.0)
    query = "What are its advantages versus alternatives?"
    memory.add_memory("combined", "1", query, "The prior turn discussed the topic's advantages.")
    retrieval = RetrievalEngine(
        RetrievalConfig(retrieval_top_k=1, final_top_n=1, relevance_threshold=0.0),
        dense_retriever=FakeDenseRetriever(),
    )
    generator = CapturingGenerator()
    result = CAROPipeline(
        conversation_id="combined",
        memory_manager=memory,
        retrieval_engine=retrieval,
        generator=generator,
    ).answer_query(query, turn_id="2")

    assert result.routing.route == "RETRIEVE_AND_MEMORY"
    assert result.strategy.selected_strategy == "MEMORY_AND_RETRIEVAL"
    assert result.memory is not None and result.memory.memory_found is True
    assert result.retrieval is not None and result.retrieval.retrieval_executed is True
    assert result.generation_context["memory_context"]
    assert result.generation_context["external_evidence"][0]["text"] == "External evidence"
    assert generator.contexts[0] == result.generation_context