import numpy as np

from modules.module2_baseline_rag.embeddings import EmbeddingModel
from modules.module2_baseline_rag.generator import Generator
from modules.module2_baseline_rag.pipeline import BaselineRAGPipeline
from modules.module2_baseline_rag.retriever import DenseRetriever
from modules.module2_baseline_rag.vector_store import FAISSVectorStore


def _sample_documents():
    return [
        {
            "doc_id": "doc_1",
            "text": "Paris is the capital city of France and is located in Europe.",
            "title": "Paris",
            "source": "HotPotQA",
            "split": "train",
            "metadata": {"is_supporting": True, "title": "Paris"},
        },
        {
            "doc_id": "doc_2",
            "text": "The Eiffel Tower is a landmark in Paris, France.",
            "title": "Eiffel Tower",
            "source": "HotPotQA",
            "split": "train",
            "metadata": {"is_supporting": False, "title": "Eiffel Tower"},
        },
        {
            "doc_id": "doc_3",
            "text": "Berlin is the capital of Germany and is in Europe.",
            "title": "Berlin",
            "source": "HotPotQA",
            "split": "train",
            "metadata": {"is_supporting": True, "title": "Berlin"},
        },
    ]


def test_embedding_model_loads():
    model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector = model.encode_query("What is the capital of France?")
    assert vector is not None
    assert vector.shape[0] > 0


def test_documents_can_be_embedded():
    model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [doc["text"] for doc in _sample_documents()]
    vectors = model.encode_documents(docs)
    assert vectors.shape[0] == len(docs)
    assert vectors.shape[1] > 0


def test_faiss_index_can_be_created_and_saved(tmp_path):
    index_path = tmp_path / "hotpotqa_index.faiss"
    vectors = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.4],
        [0.5, 0.4, 0.1],
    ], dtype="float32")
    store = FAISSVectorStore(index_path=index_path)
    store.build(vectors=vectors, documents=_sample_documents())
    assert store.index is not None
    assert index_path.exists() is False
    store.save()
    assert index_path.exists()


def test_faiss_index_can_be_loaded(tmp_path):
    index_path = tmp_path / "hotpotqa_index.faiss"
    docs = _sample_documents()
    vectors = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.4],
        [0.5, 0.4, 0.1],
    ], dtype="float32")
    store = FAISSVectorStore(index_path=index_path)
    store.build(vectors=vectors, documents=docs)
    store.save()

    loaded = FAISSVectorStore(index_path=index_path)
    loaded.load()
    assert loaded.index is not None
    assert loaded.documents is not None
    assert len(loaded.documents) == len(docs)


def test_query_embedding_works():
    model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector = model.encode_query("capital of France")
    assert vector.shape[0] > 0


def test_retrieval_returns_top_k_documents(tmp_path):
    index_path = tmp_path / "hotpotqa_index.faiss"
    docs = _sample_documents()
    vectors = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.4],
        [0.5, 0.4, 0.1],
    ], dtype="float32")
    store = FAISSVectorStore(index_path=index_path)
    store.build(vectors=vectors, documents=docs)
    retriever = DenseRetriever(store)
    results = retriever.search(np.array([0.15, 0.18, 0.32], dtype="float32"), top_k=2)
    assert len(results) == 2
    for item in results:
        assert "text" in item
        assert "metadata" in item


def test_generator_receives_query_and_context():
    generator = Generator(model_name="dummy-model")
    prompt = generator.build_prompt("What is Paris?", ["Paris is the capital of France."])
    assert "What is Paris?" in prompt
    assert "Paris is the capital of France." in prompt


def test_end_to_end_baseline_rag_query_works(tmp_path):
    docs = _sample_documents()
    vectors = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.4],
        [0.5, 0.4, 0.1],
    ], dtype="float32")
    store = FAISSVectorStore(index_path=tmp_path / "rag_index.faiss")
    store.build(vectors=vectors, documents=docs)
    retriever = DenseRetriever(store)
    generator = Generator(model_name="dummy-model")
    generator.generate = lambda query, context: f"Paris is the capital of France. Context used: {len(context)} documents."

    class TestEmbeddingModel:
        def encode_query(self, query):
            return np.array([0.15, 0.18, 0.32], dtype="float32")

    pipeline = BaselineRAGPipeline(
        vector_store=store,
        retriever=retriever,
        generator=generator,
        embedding_model=TestEmbeddingModel(),
        top_k=2,
    )

    answer = pipeline.answer_query("What is the capital of France?")
    assert "Paris" in answer
