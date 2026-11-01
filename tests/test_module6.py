import numpy as np

from modules.module3_adaptive_query_router.router import AdaptiveQueryRouter
from modules.module4_retrieval_strategy.strategy_selector import RetrievalStrategySelector
from modules.module6_temporary_memory.memory_manager import TemporaryMemoryManager


class FakeEmbedder:
    vectors = {
        "rest": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "python": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "database": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    def encode_query(self, text):
        lowered = text.lower()
        if "rest" in lowered or "api" in lowered or "advantages" in lowered:
            return self.vectors["rest"]
        if "python" in lowered:
            return self.vectors["python"]
        return self.vectors["database"]


def manager(tmp_path, **kwargs):
    return TemporaryMemoryManager(tmp_path, embedder=FakeEmbedder(), **kwargs)


def test_store_and_retrieve_one_memory(tmp_path):
    memory = manager(tmp_path)
    memory.add_memory("conversation-a", "1", "What is REST API?", "REST API is a web service style.", "REST context")
    result = memory.retrieve_memory("What are its advantages?", "conversation-a")
    assert result.memory_found is True
    assert result.memories[0].turn_id == "1"
    assert result.memories[0].stored_context == "REST context"
    assert result.memories[0].retrieval_rank == 1


def test_most_relevant_memory_and_top_k(tmp_path):
    memory = manager(tmp_path, memory_top_k=1)
    memory.add_memory("c", "1", "What is REST API?", context="REST")
    memory.add_memory("c", "2", "What is Python?", context="Python")
    result = memory.retrieve_memory("What are its advantages?", "c")
    assert len(result.memories) == 1
    assert result.memories[0].context if hasattr(result.memories[0], "context") else result.memories[0].stored_context == "REST"


def test_threshold_and_empty_memory(tmp_path):
    memory = manager(tmp_path, memory_relevance_threshold=1.1)
    memory.add_memory("c", "1", "What is REST API?", context="REST")
    result = memory.retrieve_memory("What are its advantages?", "c")
    assert result.memory_found is False
    assert result.memories == []
    empty = manager(tmp_path / "empty").retrieve_memory("anything", "c")
    assert empty.memory_found is False


def test_conversation_isolation_and_clear(tmp_path):
    memory = manager(tmp_path)
    memory.add_memory("a", "1", "What is REST API?", context="REST")
    assert memory.retrieve_memory("What are its advantages?", "b").memory_found is False
    assert memory.clear_memory("a") == 1
    assert memory.retrieve_memory("What are its advantages?", "a").memory_found is False


def test_deterministic_retrieval_and_metadata(tmp_path):
    memory = manager(tmp_path, memory_top_k=2)
    memory.add_memory("c", "2", "What is REST API?", metadata={"source": "test"})
    memory.add_memory("c", "1", "What is REST API?", metadata={"source": "test"})
    first = memory.retrieve_memory("What are its advantages?", "c").to_dict()
    second = memory.retrieve_memory("What are its advantages?", "c").to_dict()
    assert first == second
    assert first["memories"][0]["conversation_id"] == "c"
    assert first["memories"][0]["similarity"] == 1.0


def test_python_context_example(tmp_path):
    memory = manager(tmp_path)
    memory.add_memory("python-session", "1", "What is Python?", "Python is a programming language.", "Python context")
    result = memory.retrieve_memory("What are its advantages?", "python-session")
    assert result.memory_found is True
    assert result.memories[0].stored_context == "Python context"


def test_m3_m4_memory_compatibility(tmp_path):
    routing = AdaptiveQueryRouter().route("What about its advantages?")
    strategy = RetrievalStrategySelector().select(routing)
    assert strategy.selected_strategy == "MEMORY"
    memory = manager(tmp_path)
    memory.add_memory("c", "1", "What is REST API?", context="REST")
    result = memory.retrieve_memory(routing.query, "c", strategy=strategy.selected_strategy)
    assert result.memory_found is True


def test_memory_and_retrieval_only_reads_memory(tmp_path):
    memory = manager(tmp_path)
    memory.add_memory("c", "1", "What is REST API?", context="REST")
    result = memory.retrieve_memory("What are its advantages?", "c", strategy="MEMORY_AND_RETRIEVAL")
    assert result.memory_found is True
    assert not hasattr(result, "documents")
