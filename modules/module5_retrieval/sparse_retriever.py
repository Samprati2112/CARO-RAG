from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import RetrievalConfig

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(str(text).lower())


class SparseRetriever:
    """Deterministic, memory-efficient CSR BM25 retrieval."""

    _memory_cache: dict[str, tuple[str, "SparseRetriever"]] = {}

    def __init__(self, documents_path: str | Path, config: RetrievalConfig | None = None):
        self.documents_path = Path(documents_path)
        self.config = config or RetrievalConfig()
        self.k1 = 1.5
        self.b = 0.75
        self.documents: list[dict[str, Any]] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_lengths: np.ndarray = np.empty(0, dtype=np.int32)
        self.postings_doc_ids: np.ndarray = np.empty(0, dtype=np.int32)
        self.postings_tfs: np.ndarray = np.empty(0, dtype=np.int16)
        self.offsets: np.ndarray = np.empty(0, dtype=np.int64)
        self.vocab: list[str] = []
        self.term_to_id: dict[str, int] = {}
        self.term_document_frequency: dict[str, int] = {}
        self.average_document_length = 0.0
        self._load_or_build()

    def _fingerprint(self, target_cache_path: Path | None = None) -> str:
        stat = self.documents_path.stat()
        cache_path = target_cache_path or self._cache_path()
        max_docs_tag = "all" if self.config.max_documents is None else str(self.config.max_documents)
        dataset_tag = getattr(self.config, "dataset", "") or self.documents_path.stem
        key = (
            f"{self.documents_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:"
            f"{max_docs_tag}:{cache_path.resolve()}:{dataset_tag}:{self.k1}:{self.b}"
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _cache_path(self) -> Path:
        if self.config.sparse_cache_path:
            return Path(self.config.sparse_cache_path)
        doc_name = self.documents_path.name.lower()
        dataset_name = (getattr(self.config, "dataset", "") or "").lower()
        is_qrecc = ("qrecc" in doc_name) or (dataset_name == "qrecc")
        is_full = (self.config.max_documents is None) or (self.config.max_documents > 50000) or (getattr(self.config, "index_type", "dev") == "full")
        if is_qrecc:
            if is_full:
                from config.config import QRECC_FULL_BM25_PATH
                return QRECC_FULL_BM25_PATH
            from config.config import QRECC_DEV_BM25_PATH
            return QRECC_DEV_BM25_PATH
        if is_full:
            from config.config import HOTPOTQA_FULL_BM25_PATH, HOTPOTQA_INDEX_DIR
            hotpot_npz = HOTPOTQA_INDEX_DIR / "bm25_cache.npz"
            if hotpot_npz.exists() and "dummy" not in doc_name:
                return hotpot_npz
            return HOTPOTQA_FULL_BM25_PATH
        return self.documents_path.parent / "hotpotqa_dev_bm25.pkl"

    def _load_npz(self, path: Path, expected_fp: str) -> bool:
        try:
            data = np.load(path, allow_pickle=True)
            cached_fp = str(data["fingerprint"]) if "fingerprint" in data else ""
            if cached_fp and cached_fp != expected_fp:
                return False
            self.postings_doc_ids = np.asarray(data["postings_doc_ids"], dtype=np.int32)
            self.postings_tfs = np.asarray(data["postings_tfs"], dtype=np.int16)
            self.offsets = np.asarray(data["offsets"], dtype=np.int64)
            self.vocab = [str(x) for x in data["vocab"]]
            self.term_to_id = {t: idx for idx, t in enumerate(self.vocab)}
            self.doc_lengths = np.asarray(data["doc_lengths"], dtype=np.int32)
            self.average_document_length = float(data["average_document_length"])
            
            doc_ids = data["doc_ids"]
            doc_titles = data["doc_titles"]
            doc_texts = data["doc_texts"]
            self.documents = [
                {
                    "doc_id": str(did),
                    "document_id": str(did),
                    "title": str(dt),
                    "text": str(dx),
                }
                for did, dt, dx in zip(doc_ids, doc_titles, doc_texts)
            ]
            self.term_document_frequency = {
                term: int(self.offsets[i + 1] - self.offsets[i])
                for i, term in enumerate(self.vocab)
            }
            return True
        except Exception:
            return False

    def _save_npz(self, path: Path, fingerprint: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(f"{path.stem}.staging.npz")
        try:
            np.savez_compressed(
                staging,
                postings_doc_ids=self.postings_doc_ids,
                postings_tfs=self.postings_tfs,
                offsets=self.offsets,
                vocab=np.array(self.vocab, dtype=str),
                doc_lengths=self.doc_lengths,
                average_document_length=np.float64(self.average_document_length),
                doc_ids=np.array([d["doc_id"] for d in self.documents], dtype=str),
                doc_titles=np.array([d.get("title", "") for d in self.documents], dtype=str),
                doc_texts=np.array([d.get("text", "") for d in self.documents], dtype=str),
                fingerprint=np.array(fingerprint),
            )
            staging.replace(path)
        except OSError:
            if staging.exists():
                try:
                    staging.unlink()
                except OSError:
                    pass

    def _load_pkl(self, path: Path, expected_fp: str) -> bool:
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("fingerprint") != expected_fp:
                return False
            if "offsets" in payload and "postings_doc_ids" in payload:
                self.postings_doc_ids = np.asarray(payload["postings_doc_ids"], dtype=np.int32)
                self.postings_tfs = np.asarray(payload["postings_tfs"], dtype=np.int16)
                self.offsets = np.asarray(payload["offsets"], dtype=np.int64)
                self.vocab = list(payload["vocab"])
                self.term_to_id = {term: idx for idx, term in enumerate(self.vocab)}
                self.doc_lengths = np.asarray(payload["doc_lengths"], dtype=np.int32)
                self.average_document_length = float(payload["average_document_length"])
                self.documents = [
                    {
                        "doc_id": str(d.get("doc_id") or d.get("document_id") or ""),
                        "document_id": str(d.get("doc_id") or d.get("document_id") or ""),
                        "title": str(d.get("title") or ""),
                        "text": str(d.get("text") or ""),
                    }
                    for d in payload["documents"]
                ]
                self.term_document_frequency = dict(payload.get("term_document_frequency") or {})
                if not self.term_document_frequency and self.vocab:
                    self.term_document_frequency = {
                        term: int(self.offsets[i + 1] - self.offsets[i])
                        for i, term in enumerate(self.vocab)
                    }
                return True
            elif "doc_tokens" in payload:
                self.documents = [
                    {
                        "doc_id": str(d.get("doc_id") or d.get("document_id") or ""),
                        "document_id": str(d.get("doc_id") or d.get("document_id") or ""),
                        "title": str(d.get("title") or ""),
                        "text": str(d.get("text") or ""),
                    }
                    for d in payload["documents"]
                ]
                self._build_csr_from_tokens(
                    payload["doc_tokens"],
                    payload.get("doc_lengths"),
                    payload.get("average_document_length"),
                )
                return True
            return False
        except Exception:
            return False

    def _save_pkl(self, path: Path, fingerprint: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(f"{path.stem}.staging.pkl")
        try:
            with staging.open("wb") as handle:
                pickle.dump({
                    "fingerprint": fingerprint,
                    "documents": self.documents,
                    "doc_lengths": self.doc_lengths,
                    "offsets": self.offsets,
                    "postings_doc_ids": self.postings_doc_ids,
                    "postings_tfs": self.postings_tfs,
                    "vocab": self.vocab,
                    "term_document_frequency": self.term_document_frequency,
                    "average_document_length": self.average_document_length,
                }, handle, protocol=pickle.HIGHEST_PROTOCOL)
            staging.replace(path)
        except OSError:
            if staging.exists():
                try:
                    staging.unlink()
                except OSError:
                    pass

    def _build_csr_from_tokens(self, doc_tokens: list[list[str]], doc_lengths_raw=None, avg_len=None) -> None:
        if doc_lengths_raw is not None:
            self.doc_lengths = np.asarray(doc_lengths_raw, dtype=np.int32)
        else:
            self.doc_lengths = np.array([len(t) for t in doc_tokens], dtype=np.int32)
        
        if avg_len is not None:
            self.average_document_length = float(avg_len)
        else:
            total = int(self.doc_lengths.sum()) if len(self.doc_lengths) > 0 else 0
            self.average_document_length = float(total / len(self.doc_lengths)) if len(self.doc_lengths) > 0 else 0.0

        df_counter: Counter[str] = Counter()
        for tokens in doc_tokens:
            for t in set(tokens):
                df_counter[t] += 1

        self.vocab = sorted(df_counter.keys())
        self.term_to_id = {t: idx for idx, t in enumerate(self.vocab)}
        num_terms = len(self.vocab)
        self.offsets = np.zeros(num_terms + 1, dtype=np.int64)
        if num_terms > 0:
            counts = [df_counter[t] for t in self.vocab]
            np.cumsum(counts, out=self.offsets[1:])
        total_postings = int(self.offsets[-1]) if num_terms > 0 else 0
        self.postings_doc_ids = np.empty(total_postings, dtype=np.int32)
        self.postings_tfs = np.empty(total_postings, dtype=np.int16)
        if num_terms > 0:
            write_ptr = self.offsets[:-1].copy()
            for doc_idx, tokens in enumerate(doc_tokens):
                counts = Counter(tokens)
                for t, tf in counts.items():
                    tid = self.term_to_id.get(t)
                    if tid is not None:
                        pos = write_ptr[tid]
                        self.postings_doc_ids[pos] = doc_idx
                        self.postings_tfs[pos] = min(tf, 32767)
                        write_ptr[tid] += 1
        self.term_document_frequency = {
            term: int(self.offsets[i + 1] - self.offsets[i])
            for i, term in enumerate(self.vocab)
        }

    def _load_or_build(self) -> None:
        fingerprint = self._fingerprint()
        cache_path = self._cache_path()
        mem_key = f"{self.documents_path.resolve()}:{cache_path.resolve()}:{self.config.max_documents}"
        cached = self._memory_cache.get(mem_key)
        if cached and cached[0] == fingerprint:
            self.__dict__.update(cached[1].__dict__)
            return

        from config.config import HOTPOTQA_INDEX_DIR
        hotpot_npz_path = HOTPOTQA_INDEX_DIR / "bm25_cache.npz"

        loaded = False
        if cache_path.exists():
            if cache_path.suffix == ".npz":
                loaded = self._load_npz(cache_path, fingerprint)
            else:
                loaded = self._load_pkl(cache_path, fingerprint)
        elif hotpot_npz_path.exists() and ("hotpotqa" in str(self.documents_path).lower() or getattr(self.config, "dataset", "") == "hotpotqa") and "dummy" not in str(self.documents_path).lower():
            hotpot_fp = self._fingerprint(hotpot_npz_path)
            loaded = self._load_npz(hotpot_npz_path, hotpot_fp)
            if loaded:
                cache_path = hotpot_npz_path

        if loaded:
            self._memory_cache[mem_key] = (fingerprint, self)
            return

        self._build()

        if cache_path.suffix == ".npz":
            self._save_npz(cache_path, fingerprint)
        else:
            self._save_pkl(cache_path, fingerprint)

        self._memory_cache[mem_key] = (fingerprint, self)

    def _build(self) -> None:
        if not self.documents_path.exists():
            raise FileNotFoundError(f"Processed documents not found at {self.documents_path}")
        
        # Load metadata once, dropping unused fields (keep doc_id, title, text)
        documents: list[dict[str, Any]] = []
        doc_lengths_list: list[int] = []
        df_counter: Counter[str] = Counter()

        with self.documents_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    document = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = document.get("text") or document.get("content") or document.get("passage")
                if not text or not str(text).strip():
                    continue
                doc_id = str(document.get("doc_id") or document.get("document_id") or document.get("id") or len(documents))
                title = str(document.get("title") or "")
                text_str = str(text)

                clean_doc = {
                    "doc_id": doc_id,
                    "document_id": doc_id,
                    "title": title,
                    "text": text_str,
                }
                documents.append(clean_doc)

                tokens = tokenize(text_str)
                doc_lengths_list.append(len(tokens))
                for term in set(tokens):
                    df_counter[term] += 1

                if self.config.max_documents is not None and len(documents) >= self.config.max_documents:
                    break

        self.documents = documents
        self.doc_lengths = np.array(doc_lengths_list, dtype=np.int32)
        total_tokens = int(self.doc_lengths.sum()) if len(self.doc_lengths) > 0 else 0
        self.average_document_length = float(total_tokens / len(self.doc_lengths)) if len(self.doc_lengths) > 0 else 0.0

        # Build CSR postings (doc_ids int32, tfs int16) per term with CSR-style offsets
        self.vocab = sorted(df_counter.keys())
        self.term_to_id = {term: idx for idx, term in enumerate(self.vocab)}
        num_terms = len(self.vocab)
        self.offsets = np.zeros(num_terms + 1, dtype=np.int64)
        if num_terms > 0:
            counts = [df_counter[t] for t in self.vocab]
            np.cumsum(counts, out=self.offsets[1:])
        
        total_postings = int(self.offsets[-1]) if num_terms > 0 else 0
        self.postings_doc_ids = np.empty(total_postings, dtype=np.int32)
        self.postings_tfs = np.empty(total_postings, dtype=np.int16)

        if num_terms > 0:
            write_ptr = self.offsets[:-1].copy()
            for doc_idx, doc in enumerate(self.documents):
                tokens = tokenize(doc["text"])
                counts = Counter(tokens)
                for term, tf in counts.items():
                    tid = self.term_to_id.get(term)
                    if tid is not None:
                        pos = write_ptr[tid]
                        self.postings_doc_ids[pos] = doc_idx
                        self.postings_tfs[pos] = min(tf, 32767)
                        write_ptr[tid] += 1

        self.term_document_frequency = {
            term: int(self.offsets[i + 1] - self.offsets[i])
            for i, term in enumerate(self.vocab)
        }

    @classmethod
    def build_bm25_index(
        cls,
        documents_path: str | Path,
        cache_path: str | Path | None = None,
        *,
        max_documents: int | None = None,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        """Explicit builder for independent BM25 full or development index."""
        if cache_path is None:
            ds = (dataset_name or Path(documents_path).stem).lower()
            if "hotpotqa" in ds and "dummy" not in Path(documents_path).name.lower():
                from config.config import HOTPOTQA_INDEX_DIR
                cache_path = HOTPOTQA_INDEX_DIR / "bm25_cache.npz"
        cfg = RetrievalConfig(
            max_documents=max_documents,
            sparse_cache_path=str(cache_path) if cache_path else None,
            dataset=dataset_name or Path(documents_path).stem,
            index_type="full" if max_documents is None else "dev",
        )
        retriever = cls(documents_path, config=cfg)
        return {
            "dataset": dataset_name or Path(documents_path).stem,
            "documents_indexed": len(retriever.documents),
            "cache_path": str(retriever._cache_path()),
            "average_doc_length": retriever.average_document_length,
            "vocabulary_size": len(retriever.term_document_frequency),
            "max_documents": max_documents,
        }

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.documents:
            return []
        query_terms = set(query_tokens)
        document_count = len(self.documents)
        k = top_k or self.config.retrieval_top_k

        scores = np.zeros(document_count, dtype=np.float64)
        for term in query_terms:
            tid = self.term_to_id.get(term)
            if tid is None:
                continue
            start = self.offsets[tid]
            end = self.offsets[tid + 1]
            if start == end:
                continue
            matching_docs = self.postings_doc_ids[start:end]
            tfs = self.postings_tfs[start:end]
            df = int(end - start)
            idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            doc_len = self.doc_lengths[matching_docs]
            length_norm = 1.0 - self.b + self.b * (doc_len / self.average_document_length)
            term_scores = idf * (tfs * (self.k1 + 1.0)) / (tfs + self.k1 * length_norm)
            scores[matching_docs] += term_scores

        matching_indices = np.where(scores > 0)[0]
        scored: list[tuple[float, int]] = [(float(scores[i]), int(i)) for i in matching_indices]
        scored.sort(key=lambda item: (-item[0], item[1]))

        if len(scored) < k:
            seen_indices = set(matching_indices)
            for idx in range(document_count):
                if idx not in seen_indices:
                    scored.append((0.0, idx))
                    if len(scored) >= k:
                        break

        results = []
        for score, index in scored[:k]:
            document = dict(self.documents[index])
            document["retrieval_score"] = float(score)
            document["sparse_score"] = float(score)
            document["retrieval_rank"] = len(results) + 1
            document["normalized_score"] = float(document["retrieval_score"])
            results.append(document)
        return results
