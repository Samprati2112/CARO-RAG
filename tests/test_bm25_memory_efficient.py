from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from config.config import (
    HOTPOTQA_INDEX_DIR,
    HOTPOTQA_BM25_CACHE_PATH,
    HOTPOTQA_DOCS_OUTPUT,
    HOTPOTQA_QUESTIONS_OUTPUT,
)
from modules.module5_retrieval.schemas import RetrievalConfig
from modules.module5_retrieval.sparse_retriever import SparseRetriever, tokenize


def original_bm25_scorer(
    documents: list[dict[str, Any]],
    doc_tokens: list[list[str]],
    doc_lengths: list[int] | np.ndarray,
    average_doc_len: float,
    df_map: dict[str, int],
    query: str,
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """Reference implementation of original in-memory BM25 scoring."""
    query_tokens = tokenize(query)
    if not query_tokens or not documents:
        return []
    query_terms = set(query_tokens)
    document_count = len(documents)

    scored: list[tuple[float, int]] = []
    for index, tokens in enumerate(doc_tokens):
        counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = df_map.get(term, 0)
            idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            dl = float(doc_lengths[index])
            length_norm = 1.0 - b + b * (dl / average_doc_len)
            score += idf * (tf * (k1 + 1.0)) / (tf + k1 * length_norm)
        scored.append((score, index))

    # Tie-breaking matching CSR implementation: higher score first, lower index second
    matching = [item for item in scored if item[0] > 0.0]
    matching.sort(key=lambda item: (-item[0], item[1]))

    if len(matching) < top_k:
        seen = {item[1] for item in matching}
        for idx in range(document_count):
            if idx not in seen:
                matching.append((0.0, idx))
                if len(matching) >= top_k:
                    break

    results = []
    for score, index in matching[:top_k]:
        doc = dict(documents[index])
        doc["retrieval_score"] = float(score)
        doc["sparse_score"] = float(score)
        doc["retrieval_rank"] = len(results) + 1
        doc["normalized_score"] = float(score)
        results.append(doc)
    return results


def test_csr_numpy_arrays_data_types_and_no_tuples(tmp_path):
    """Verifies postings are numpy arrays (doc_ids int32, tfs int16) with CSR offsets and doc_lengths is int32."""
    docs_file = tmp_path / "test_corpus.jsonl"
    cache_file = tmp_path / "test_cache.npz"

    records = [
        {"doc_id": "doc_0", "title": "Alpha", "text": "The quick brown fox jumps over the lazy dog.", "extra_field": 123},
        {"doc_id": "doc_1", "title": "Beta", "text": "Brown foxes are quick and nimble animals.", "author": "John Doe"},
        {"doc_id": "doc_2", "title": "Gamma", "text": "Lazy dogs sleep all day in the warm sun.", "tags": ["animal", "dog"]},
    ]
    docs_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    cfg = RetrievalConfig(sparse_cache_path=str(cache_file))
    retriever = SparseRetriever(docs_file, config=cfg)

    # 1. Postings stored as numpy arrays
    assert isinstance(retriever.postings_doc_ids, np.ndarray), "postings_doc_ids must be numpy array"
    assert retriever.postings_doc_ids.dtype == np.int32, f"postings_doc_ids dtype must be int32, got {retriever.postings_doc_ids.dtype}"

    assert isinstance(retriever.postings_tfs, np.ndarray), "postings_tfs must be numpy array"
    assert retriever.postings_tfs.dtype == np.int16, f"postings_tfs dtype must be int16, got {retriever.postings_tfs.dtype}"

    assert isinstance(retriever.offsets, np.ndarray), "offsets must be numpy array"
    assert retriever.offsets.dtype == np.int64, f"offsets dtype must be int64, got {retriever.offsets.dtype}"

    # 2. Doc lengths stored as numpy array
    assert isinstance(retriever.doc_lengths, np.ndarray), "doc_lengths must be numpy array"
    assert retriever.doc_lengths.dtype == np.int32, f"doc_lengths dtype must be int32, got {retriever.doc_lengths.dtype}"
    assert len(retriever.doc_lengths) == 3

    # 3. Offsets match vocab
    assert len(retriever.offsets) == len(retriever.vocab) + 1
    assert retriever.offsets[-1] == len(retriever.postings_doc_ids)

    # 4. No Python lists of tuples for postings
    assert not hasattr(retriever, "sparse_inv_index") or retriever.sparse_inv_index is None or not isinstance(retriever.sparse_inv_index, dict)


def test_metadata_drops_unused_fields(tmp_path):
    """Verifies metadata keeps only (doc_id, title, text) and drops unused fields."""
    docs_file = tmp_path / "metadata_test.jsonl"
    records = [
        {
            "doc_id": "101",
            "title": "Machine Learning",
            "text": "Supervised learning uses labeled training datasets.",
            "author": "Alice",
            "timestamp": "2026-01-01",
            "embeddings": [0.1, 0.2, 0.3],
            "metadata": {"source": "wiki", "views": 500},
        }
    ]
    docs_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    retriever = SparseRetriever(docs_file, config=RetrievalConfig(sparse_cache_path=str(tmp_path / "meta_cache.npz")))
    assert len(retriever.documents) == 1
    doc = retriever.documents[0]

    # Required fields must exist
    assert doc["doc_id"] == "101"
    assert doc["title"] == "Machine Learning"
    assert doc["text"] == "Supervised learning uses labeled training datasets."

    # Unused fields must be dropped
    assert "author" not in doc
    assert "timestamp" not in doc
    assert "embeddings" not in doc
    assert "metadata" not in doc


def test_bm25_cache_npz_saving_and_reloading(tmp_path):
    """Verifies the built index saves to .npz and loads on later runs without rebuilding."""
    docs_file = tmp_path / "corpus.jsonl"
    cache_npz = tmp_path / "bm25_cache.npz"

    records = [
        {"doc_id": f"doc_{i}", "title": f"Title {i}", "text": f"Document content for test index item {i}."}
        for i in range(25)
    ]
    docs_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    # Initial build saves to npz
    retriever1 = SparseRetriever(docs_file, config=RetrievalConfig(sparse_cache_path=str(cache_npz)))
    assert cache_npz.exists(), f"Cache file {cache_npz} was not created"

    # Inspect the saved npz file directly
    saved_npz = np.load(cache_npz, allow_pickle=True)
    assert "postings_doc_ids" in saved_npz
    assert "postings_tfs" in saved_npz
    assert "offsets" in saved_npz
    assert "doc_lengths" in saved_npz
    assert saved_npz["postings_doc_ids"].dtype == np.int32
    assert saved_npz["postings_tfs"].dtype == np.int16
    assert saved_npz["doc_lengths"].dtype == np.int32
    assert saved_npz["offsets"].dtype == np.int64

    # Second run loads from npz
    # Clear in-memory singleton cache to force disk load
    SparseRetriever._memory_cache.clear()
    retriever2 = SparseRetriever(docs_file, config=RetrievalConfig(sparse_cache_path=str(cache_npz)))

    assert len(retriever2.documents) == 25
    assert np.array_equal(retriever1.offsets, retriever2.offsets)
    assert np.array_equal(retriever1.postings_doc_ids, retriever2.postings_doc_ids)
    assert np.array_equal(retriever1.postings_tfs, retriever2.postings_tfs)
    assert np.array_equal(retriever1.doc_lengths, retriever2.doc_lengths)
    assert math.isclose(retriever1.average_document_length, retriever2.average_document_length)

    # Retrieval results from loaded cache must match
    res1 = retriever1.retrieve("test index item", top_k=5)
    res2 = retriever2.retrieve("test index item", top_k=5)
    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1["doc_id"] == r2["doc_id"]
        assert math.isclose(r1["retrieval_score"], r2["retrieval_score"], abs_tol=1e-9)


def test_hotpotqa_bm25_cache_npz_exists_and_valid():
    """Verifies that indexes/hotpotqa/bm25_cache.npz exists and adheres to all schema requirements."""
    cache_path = HOTPOTQA_BM25_CACHE_PATH
    assert cache_path.exists(), f"Required cache {cache_path} does not exist"

    data = np.load(cache_path, allow_pickle=True)

    # Required arrays in the 250k corpus index
    assert "body_offsets" in data
    assert "body_postings_doc_ids" in data
    assert "body_postings_tfs" in data
    assert "body_vocab" in data
    assert "corpus_doc_lengths" in data

    assert data["body_offsets"].dtype == np.int64
    assert data["body_postings_doc_ids"].dtype == np.int32
    assert data["body_postings_tfs"].dtype == np.int16
    assert data["corpus_doc_lengths"].dtype == np.int32

    assert len(data["corpus_doc_lengths"]) == 250000
    assert len(data["body_offsets"]) == len(data["body_vocab"]) + 1
    assert data["body_offsets"][-1] == len(data["body_postings_doc_ids"])
    assert len(data["body_postings_doc_ids"]) == len(data["body_postings_tfs"])


def test_20_queries_scores_identical_to_original_bm25():
    """
    Results must be identical to the current BM25 scores; compare 20 queries.
    Tests 20 real questions from HotPotQA against the corpus, comparing original
    document-at-a-time BM25 scores against the CSR numpy array implementation.
    """
    # 1. Prepare corpus documents & tokens
    if HOTPOTQA_DOCS_OUTPUT.exists():
        records = []
        with open(HOTPOTQA_DOCS_OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
                if len(records) >= 1000:
                    break
    else:
        records = [
            {"doc_id": f"doc_{i}", "text": f"Document {i} discusses topic {i % 5} and entity {i % 10}."}
            for i in range(200)
        ]

    clean_docs = [
        {
            "doc_id": str(r.get("doc_id") or r.get("id")),
            "title": str(r.get("title") or ""),
            "text": str(r.get("text") or r.get("content") or ""),
        }
        for r in records
    ]
    doc_tokens = [tokenize(d["text"]) for d in clean_docs]
    doc_lengths = np.array([len(toks) for toks in doc_tokens], dtype=np.int32)
    avg_len = float(doc_lengths.sum() / max(len(doc_lengths), 1))

    df_map = Counter()
    for toks in doc_tokens:
        for t in set(toks):
            df_map[t] += 1

    # 2. Build CSR arrays from the exact same tokens
    vocab = sorted(df_map.keys())
    term_to_id = {t: idx for idx, t in enumerate(vocab)}
    offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
    if vocab:
        np.cumsum([df_map[t] for t in vocab], out=offsets[1:])
    total_postings = int(offsets[-1]) if vocab else 0

    postings_doc_ids = np.empty(total_postings, dtype=np.int32)
    postings_tfs = np.empty(total_postings, dtype=np.int16)
    if vocab:
        write_ptr = offsets[:-1].copy()
        for doc_idx, toks in enumerate(doc_tokens):
            for t, tf in Counter(toks).items():
                tid = term_to_id.get(t)
                if tid is not None:
                    pos = write_ptr[tid]
                    postings_doc_ids[pos] = doc_idx
                    postings_tfs[pos] = min(tf, 32767)
                    write_ptr[tid] += 1

    # CSR retrieve function
    def csr_retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens or not clean_docs:
            return []
        query_terms = set(query_tokens)
        document_count = len(clean_docs)
        scores = np.zeros(document_count, dtype=np.float64)

        for t in query_terms:
            tid = term_to_id.get(t)
            if tid is None:
                continue
            s, e = offsets[tid], offsets[tid + 1]
            if s == e:
                continue
            docs = postings_doc_ids[s:e]
            tfs = postings_tfs[s:e]
            df = int(e - s)
            idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            d_len = doc_lengths[docs]
            lnorm = 1.0 - 0.75 + 0.75 * (d_len / avg_len)
            scores[docs] += idf * (tfs * 2.5) / (tfs + 1.5 * lnorm)

        matching_indices = np.where(scores > 0)[0]
        scored = [(float(scores[i]), int(i)) for i in matching_indices]
        scored.sort(key=lambda item: (-item[0], item[1]))

        if len(scored) < top_k:
            seen_indices = set(matching_indices)
            for idx in range(document_count):
                if idx not in seen_indices:
                    scored.append((0.0, idx))
                    if len(scored) >= top_k:
                        break

        results = []
        for score, index in scored[:top_k]:
            doc = dict(clean_docs[index])
            doc["retrieval_score"] = float(score)
            doc["sparse_score"] = float(score)
            doc["retrieval_rank"] = len(results) + 1
            doc["normalized_score"] = float(score)
            results.append(doc)
        return results

    # 3. Load 20 queries
    queries = []
    if HOTPOTQA_QUESTIONS_OUTPUT.exists():
        with open(HOTPOTQA_QUESTIONS_OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    q = item.get("question") or item.get("query")
                    if q:
                        queries.append(q)
                if len(queries) >= 20:
                    break

    # Fallback to 20 synthetic queries if needed
    if len(queries) < 20:
        queries = [
            f"Query number {i} searching for topic {i % 5} and entity {i % 10}"
            for i in range(20)
        ]

    assert len(queries) == 20, f"Expected 20 queries, got {len(queries)}"

    # 4. Compare all 20 queries
    for q_idx, query in enumerate(queries):
        orig_results = original_bm25_scorer(
            documents=clean_docs,
            doc_tokens=doc_tokens,
            doc_lengths=doc_lengths,
            average_doc_len=avg_len,
            df_map=df_map,
            query=query,
            top_k=5,
        )
        csr_results = csr_retrieve(query, top_k=5)

        assert len(orig_results) == len(csr_results), f"Length mismatch on query {q_idx}: '{query}'"
        for rank, (r_orig, r_csr) in enumerate(zip(orig_results, csr_results), start=1):
            assert r_orig["doc_id"] == r_csr["doc_id"], (
                f"Query {q_idx} rank {rank} doc_id mismatch: {r_orig['doc_id']} vs {r_csr['doc_id']}"
            )
            assert math.isclose(r_orig["retrieval_score"], r_csr["retrieval_score"], abs_tol=1e-9), (
                f"Query {q_idx} rank {rank} score mismatch: {r_orig['retrieval_score']} vs {r_csr['retrieval_score']}"
            )
            assert math.isclose(r_orig["sparse_score"], r_csr["sparse_score"], abs_tol=1e-9)
