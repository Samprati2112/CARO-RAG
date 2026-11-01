from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pytest

from config.config import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_INDEXING_CHUNK_SIZE,
    HOTPOTQA_DEV_BM25_PATH,
    HOTPOTQA_DEV_INDEX_PATH,
    HOTPOTQA_DOCS_OUTPUT,
    HOTPOTQA_FULL_BM25_PATH,
    HOTPOTQA_FULL_INDEX_META_PATH,
    HOTPOTQA_FULL_INDEX_PATH,
    HOTPOTQA_INDEX_PATH,
    MAX_INDEX_DOCUMENTS,
    QRECC_DEV_BM25_PATH,
    QRECC_DEV_INDEX_PATH,
    QRECC_DOCS_OUTPUT,
    QRECC_FULL_BM25_PATH,
    QRECC_FULL_INDEX_META_PATH,
    QRECC_FULL_INDEX_PATH,
)
from modules.module2_baseline_rag.vector_store import FAISSVectorStore
from modules.module5_retrieval.retriever import RetrievalEngine
from modules.module5_retrieval.schemas import RetrievalConfig
from modules.module5_retrieval.sparse_retriever import SparseRetriever


class DummyEmbeddingModel:
    """Lightweight test dummy producing deterministic float32 vectors."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode_documents(self, documents, batch_size: int = 32, show_progress_bar: bool = False):
        if not documents:
            return np.empty((0, self.dim), dtype=np.float32)
        rng = np.random.default_rng(seed=42)
        matrix = rng.standard_normal((len(documents), self.dim), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def encode_query(self, query: str):
        rng = np.random.default_rng(seed=100)
        vec = rng.standard_normal(self.dim, dtype=np.float32)
        return vec / max(float(np.linalg.norm(vec)), 1e-12)


def test_config_paths_and_defaults():
    # Development paths preserved
    assert HOTPOTQA_DEV_INDEX_PATH == HOTPOTQA_INDEX_PATH
    assert HOTPOTQA_DEV_BM25_PATH.name == "hotpotqa_dev_bm25.pkl"
    assert MAX_INDEX_DOCUMENTS == 5000

    # Full index paths separated and isolated
    assert HOTPOTQA_FULL_INDEX_PATH.name == "hotpotqa_full_index.faiss"
    assert HOTPOTQA_FULL_INDEX_META_PATH.name == "hotpotqa_full_index.meta.json"
    assert HOTPOTQA_FULL_BM25_PATH.name == "hotpotqa_full_bm25.pkl"

    assert QRECC_FULL_INDEX_PATH.name == "qrecc_full_index.faiss"
    assert QRECC_FULL_INDEX_META_PATH.name == "qrecc_full_index.meta.json"
    assert QRECC_FULL_BM25_PATH.name == "qrecc_full_bm25.pkl"

    # Datasets strictly isolated in different paths
    assert HOTPOTQA_FULL_INDEX_PATH != QRECC_FULL_INDEX_PATH
    assert HOTPOTQA_FULL_BM25_PATH != QRECC_FULL_BM25_PATH
    assert "hotpotqa" in str(HOTPOTQA_FULL_INDEX_PATH)
    assert "qrecc" in str(QRECC_FULL_INDEX_PATH)

    # Defaults
    assert DEFAULT_EMBEDDING_BATCH_SIZE == 32
    assert DEFAULT_INDEXING_CHUNK_SIZE == 1000


def test_faiss_vector_store_build_from_jsonl(tmp_path):
    jsonl_file = tmp_path / "test_docs.jsonl"
    index_file = tmp_path / "test_full.faiss"

    records = [
        {"doc_id": "doc_1", "text": "First passage about history.", "title": "History", "source": "HotPotQA", "metadata": {"is_supporting": True}},
        {"doc_id": "doc_2", "text": "Second passage about geography.", "title": "Geography", "source": "HotPotQA", "metadata": {"is_supporting": False}},
        {"doc_id": "doc_3", "text": "Third passage about science.", "title": "Science", "source": "HotPotQA", "metadata": {"is_supporting": True}},
        {"doc_id": "doc_4", "text": "", "title": "Empty doc", "source": "HotPotQA"},  # empty: should be skipped
        {"doc_id": "doc_5", "text": "Fifth passage about arts.", "title": "Arts", "source": "HotPotQA", "metadata": {}},
    ]
    with jsonl_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    store = FAISSVectorStore(index_path=index_file)
    dummy_model = DummyEmbeddingModel(dim=8)

    report = store.build_from_jsonl(
        jsonl_path=jsonl_file,
        embedding_model=dummy_model,
        chunk_size=2,
        batch_size=2,
        dataset_name="hotpotqa_test",
    )

    assert report["valid_indexed_documents"] == 4
    assert report["skipped_records"] == 1
    assert report["input_records"] == 5
    assert index_file.exists()
    assert index_file.with_suffix(".meta.json").exists()

    # Staging artifacts must NOT be left behind
    assert list(tmp_path.glob("*.staging.*")) == []

    # Verify search on loaded index
    loaded = FAISSVectorStore(index_path=index_file).load()
    assert len(loaded.documents) == 4
    assert loaded.index.ntotal == 4

    query_vec = dummy_model.encode_query("science and history")
    results = loaded.search(query_vec, top_k=2)
    assert len(results) == 2
    assert "retrieval_score" in results[0]
    assert results[0]["title"] in {"History", "Geography", "Science", "Arts"}


def test_faiss_vector_store_preserves_qrecc_metadata(tmp_path):
    jsonl_file = tmp_path / "qrecc_docs.jsonl"
    index_file = tmp_path / "qrecc_full.faiss"

    records = [
        {
            "doc_id": "qrecc_1_1",
            "text": "Gary Cherone is an American singer.",
            "title": "Gary Cherone",
            "source": "QReCC",
            "conversation_id": "conv_42",
            "turn_id": 1,
            "split": "train",
            "metadata": {"question_type": "conversational"},
        },
        {
            "doc_id": "qrecc_1_2",
            "text": "Extreme is an American rock band.",
            "title": "Extreme band",
            "source": "QReCC",
            "conversation_id": "conv_42",
            "turn_id": 2,
            "split": "train",
            "metadata": {"question_type": "conversational"},
        },
    ]
    with jsonl_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    store = FAISSVectorStore(index_path=index_file)
    dummy_model = DummyEmbeddingModel(dim=8)

    store.build_from_jsonl(jsonl_file, dummy_model, chunk_size=1, batch_size=1, dataset_name="qrecc")

    loaded = FAISSVectorStore(index_path=index_file).load()
    assert len(loaded.documents) == 2
    doc = loaded.documents[0]
    assert doc["conversation_id"] == "conv_42"
    assert doc["turn_id"] == 1
    assert doc["source"] == "QReCC"


def test_faiss_vector_store_staging_clean_on_failure(tmp_path):
    jsonl_file = tmp_path / "broken_docs.jsonl"
    index_file = tmp_path / "test_abort.faiss"

    records = [{"doc_id": "1", "text": "Valid text", "title": "T"}]
    with jsonl_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    class BrokenModel:
        def encode_documents(self, documents, **kwargs):
            raise RuntimeError("Simulated GPU/OOM Failure during encoding")

    store = FAISSVectorStore(index_path=index_file)
    with pytest.raises(RuntimeError, match="Simulated GPU/OOM Failure"):
        store.build_from_jsonl(jsonl_file, BrokenModel(), chunk_size=1)

    # Neither final nor staging artifacts must exist
    assert not index_file.exists()
    assert list(tmp_path.glob("*.staging.*")) == []


def test_sparse_bm25_full_corpus_build_and_fingerprint_isolation(tmp_path):
    docs_file = tmp_path / "corpus.jsonl"
    cache_dev = tmp_path / "dev_bm25.pkl"
    cache_full = tmp_path / "full_bm25.pkl"

    records = [
        {"doc_id": f"doc_{i}", "text": f"Document number {i} discussing topic {i % 3}"}
        for i in range(10)
    ]
    docs_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    # 1. Build dev index (limit 3)
    dev_summary = SparseRetriever.build_bm25_index(
        documents_path=docs_file,
        cache_path=cache_dev,
        max_documents=3,
        dataset_name="hotpotqa_dev",
    )
    assert dev_summary["documents_indexed"] == 3
    assert cache_dev.exists()

    # 2. Build full index (no limit)
    full_summary = SparseRetriever.build_bm25_index(
        documents_path=docs_file,
        cache_path=cache_full,
        max_documents=None,
        dataset_name="hotpotqa_full",
    )
    assert full_summary["documents_indexed"] == 10
    assert cache_full.exists()

    # Verify caches are independent
    dev_retriever = SparseRetriever(
        docs_file,
        config=RetrievalConfig(sparse_cache_path=str(cache_dev), max_documents=3),
    )
    full_retriever = SparseRetriever(
        docs_file,
        config=RetrievalConfig(sparse_cache_path=str(cache_full), max_documents=None),
    )
    assert len(dev_retriever.documents) == 3
    assert len(full_retriever.documents) == 10

    # Fingerprints must be distinct
    assert dev_retriever._fingerprint() != full_retriever._fingerprint()


def test_retrieval_engine_resolves_full_vs_dev_paths(tmp_path):
    dummy_docs = tmp_path / "dummy_docs.jsonl"
    dummy_docs.write_text(json.dumps({"doc_id": "1", "text": "passage"}), encoding="utf-8")

    # Dev mode dense path
    dev_engine = RetrievalEngine(RetrievalConfig(index_type="dev", dataset="hotpotqa"))
    assert Path(dev_engine._get_dense().index_path) == HOTPOTQA_DEV_INDEX_PATH

    # Full mode HotPotQA dense path
    full_hotpot = RetrievalEngine(RetrievalConfig(index_type="full", dataset="hotpotqa", max_documents=None))
    assert Path(full_hotpot._get_dense().index_path) == HOTPOTQA_FULL_INDEX_PATH

    # Full mode QReCC dense path
    full_qrecc = RetrievalEngine(RetrievalConfig(index_type="full", dataset="qrecc", max_documents=None))
    assert Path(full_qrecc._get_dense().index_path) == QRECC_FULL_INDEX_PATH

    # Sparse cache resolution logic using controlled dummy documents path
    retriever_dev = SparseRetriever(dummy_docs, config=RetrievalConfig(index_type="dev", dataset="hotpotqa"))
    assert retriever_dev._cache_path() == dummy_docs.parent / "hotpotqa_dev_bm25.pkl"

    retriever_full_hotpot = SparseRetriever(dummy_docs, config=RetrievalConfig(index_type="full", dataset="hotpotqa", max_documents=None))
    assert retriever_full_hotpot._cache_path() == HOTPOTQA_FULL_BM25_PATH

    retriever_full_qrecc = SparseRetriever(dummy_docs, config=RetrievalConfig(index_type="full", dataset="qrecc", max_documents=None))
    assert retriever_full_qrecc._cache_path() == QRECC_FULL_BM25_PATH


def test_cli_parser_options():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hotpotqa", "qrecc", "all"], default="hotpotqa")
    p.add_argument("--module", choices=["1", "2", "3", "4", "5", "6", "7", "8"], default="1")
    p.add_argument("--action", choices=["build-index", "build-full-index", "query"], default=None)
    p.add_argument("--index-type", choices=["dev", "full"], default="dev")
    p.add_argument("--index-component", choices=["both", "dense", "sparse"], default="both")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--num-threads", type=int, default=None)
    p.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--limit", type=int, default=None)

    cmd_args = [
        "--module", "2",
        "--action", "build-full-index",
        "--dataset", "hotpotqa",
        "--index-type", "full",
        "--index-component", "dense",
        "--batch-size", "64",
        "--chunk-size", "500",
        "--num-threads", "4",
        "--no-show-progress",
        "--resume",
        "--device", "cuda",
        "--limit", "10",
    ]
    parsed = p.parse_args(cmd_args)
    assert parsed.module == "2"
    assert parsed.action == "build-full-index"
    assert parsed.dataset == "hotpotqa"
    assert parsed.index_type == "full"
    assert parsed.index_component == "dense"
    assert parsed.batch_size == 64
    assert parsed.chunk_size == 500
    assert parsed.num_threads == 4
    assert parsed.show_progress is False
    assert parsed.resume is True
    assert parsed.device == "cuda"
    assert parsed.limit == 10


def test_checkpoint_interruption_and_safe_resume(tmp_path):
    """Verifies that an interrupted indexing run can safely resume from its checkpoint without data loss or duplication."""
    jsonl_file = tmp_path / "resume_docs.jsonl"
    index_file = tmp_path / "resume_index.faiss"

    # Create 6 distinct documents
    records = [
        {"doc_id": f"doc_{i}", "text": f"Document content number {i} for testing.", "title": f"Title {i}"}
        for i in range(1, 7)
    ]
    with jsonl_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    store = FAISSVectorStore(index_path=index_file)
    dummy_model = DummyEmbeddingModel(dim=8)

    # Simulate an error during the second chunk (chunk_size=2)
    chunk_counter = {"count": 0}

    def failing_callback(*args, **kwargs):
        chunk_counter["count"] += 1
        if chunk_counter["count"] == 1:
            # Chunk 1 succeeds and checkpoints. Now abort run before chunk 2 completes.
            raise RuntimeError("Simulated crash after chunk 1 checkpoint")

    with pytest.raises(RuntimeError, match="Simulated crash"):
        store.build_from_jsonl(
            jsonl_path=jsonl_file,
            embedding_model=dummy_model,
            chunk_size=2,
            batch_size=2,
            progress_callback=failing_callback,
            resume=False,
        )

    checkpoint_index_path = index_file.with_name(f"{index_file.stem}.checkpoint.faiss")
    checkpoint_meta_path = index_file.with_name(f"{index_file.stem}.checkpoint.meta.jsonl")
    checkpoint_state_path = index_file.with_name(f"{index_file.stem}.checkpoint.state.json")

    # Verify that chunk 1 produced valid checkpoint files
    assert checkpoint_index_path.exists()
    assert checkpoint_meta_path.exists()
    assert checkpoint_state_path.exists()

    state = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))
    assert state["valid_indexed_documents"] == 2
    assert state["input_records"] == 2

    # Now resume the run
    store2 = FAISSVectorStore(index_path=index_file)
    report = store2.build_from_jsonl(
        jsonl_path=jsonl_file,
        embedding_model=dummy_model,
        chunk_size=2,
        batch_size=2,
        resume=True,
    )

    # Verify completion
    assert report["resumed"] is True
    assert report["valid_indexed_documents"] == 6
    assert index_file.exists()
    assert index_file.with_name(f"{index_file.stem}.meta.json").exists()

    # Verify all checkpoint files were cleaned up after atomic publication
    assert not checkpoint_index_path.exists()
    assert not checkpoint_meta_path.exists()
    assert not checkpoint_state_path.exists()

    # Load and verify no duplicates in vectors and metadata
    store2.load()
    assert len(store2.documents) == 6
    doc_ids = [d["doc_id"] for d in store2.documents]
    assert doc_ids == [f"doc_{i}" for i in range(1, 7)]


def test_cpu_thread_configuration():
    from modules.module2_baseline_rag.embeddings import configure_cpu_threads
    import torch

    orig_threads = torch.get_num_threads()
    try:
        configured = configure_cpu_threads(4)
        assert configured == 4
        assert torch.get_num_threads() == 4
    finally:
        configure_cpu_threads(orig_threads)

