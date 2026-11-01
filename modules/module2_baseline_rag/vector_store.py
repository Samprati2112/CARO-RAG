from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, List

import faiss
import numpy as np


class FAISSVectorStore:
    """Persistent FAISS vector store for dense retrieval."""

    def __init__(self, index_path: str | Path, *, top_k: int = 5):
        self.index_path = Path(index_path)
        self.metadata_path = self.index_path.with_suffix(".meta.json")
        self.index = None
        self.documents: List[dict] = []
        self.top_k = top_k

    def build(self, vectors: np.ndarray, documents: List[dict]):
        if len(vectors) != len(documents):
            raise ValueError(f"Vector count {len(vectors)} does not match document count {len(documents)}")
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.documents = list(documents)
        self.index.add(np.asarray(vectors, dtype=np.float32))
        return self

    def save(self):
        if self.index is None:
            raise ValueError("No FAISS index is available to save.")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        payload = [
            {**doc, "metadata": dict(doc.get("metadata", {})) if isinstance(doc.get("metadata", {}), dict) else {}}
            for doc in self.documents
        ]
        self.metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.index_path

    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                self.documents = json.load(handle)
        else:
            self.documents = []
        return self

    def search(self, query_vector: np.ndarray, top_k: int | None = None):
        if self.index is None:
            raise ValueError("FAISS index has not been built or loaded.")
        k = min(top_k or self.top_k, self.index.ntotal)
        if k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, k)
        results: List[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = dict(self.documents[int(idx)])
            doc["retrieval_score"] = float(score)
            if not isinstance(doc.get("metadata"), dict):
                doc["metadata"] = {}
            results.append(doc)
        return results

    def build_from_documents(self, documents: List[dict], embedding_model, limit: int | None = None):
        source_docs = documents[:limit] if limit is not None else documents
        texts = []
        valid_docs = []
        for doc in source_docs:
            text = doc.get("text") or doc.get("content") or doc.get("passage")
            if not text:
                continue
            metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            valid_docs.append({
                "doc_id": doc.get("doc_id") or doc.get("id") or str(len(valid_docs)),
                "text": str(text),
                "title": doc.get("title") or metadata.get("title") or "",
                "source": doc.get("source") or "unknown",
                "split": doc.get("split") or "train",
                "metadata": metadata,
                "question_id": doc.get("question_id"),
            })
            texts.append(str(text))
        if not texts:
            raise ValueError("No valid document text was found for FAISS indexing.")
        print(f"[2/5] Creating BGE-M3 embeddings for {len(texts)} documents...")
        vectors = embedding_model.encode_documents(texts)
        return self.build(vectors=vectors, documents=valid_docs)

    def build_from_jsonl(
        self,
        jsonl_path: str | Path,
        embedding_model,
        *,
        limit: int | None = None,
        chunk_size: int = 1000,
        batch_size: int = 32,
        dataset_name: str | None = None,
        progress_callback: Any = None,
        show_progress_bar: bool = False,
        resume: bool = True,
    ) -> dict[str, Any]:
        """Memory-conscious streaming builder: reads JSONL in chunks and adds vectors to FAISS.

        Features:
        - In-chunk and inter-chunk progress reporting with docs/second throughput.
        - Checkpoint and safe resume support (saves index and metadata after each chunk).
        - Staging artifacts with atomic publication upon completion.
        """
        source_path = Path(jsonl_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source JSONL file not found: {source_path}")

        source_stat = source_path.stat()
        staging_index_path = self.index_path.with_name(f"{self.index_path.stem}.staging{self.index_path.suffix}")
        staging_metadata_path = self.index_path.with_name(f"{self.index_path.stem}.staging.meta.json")

        checkpoint_index_path = self.index_path.with_name(f"{self.index_path.stem}.checkpoint.faiss")
        checkpoint_meta_path = self.index_path.with_name(f"{self.index_path.stem}.checkpoint.meta.jsonl")
        checkpoint_state_path = self.index_path.with_name(f"{self.index_path.stem}.checkpoint.state.json")

        # Clean up any stale staging artifacts from prior aborted runs
        for stale in (staging_index_path, staging_metadata_path):
            if stale.exists():
                try:
                    stale.unlink()
                except OSError:
                    pass

        self.index = None
        self.documents = []
        input_records = 0
        valid_indexed = 0
        skipped_records = 0
        resumed = False

        # Attempt to resume from checkpoint if requested
        if resume and checkpoint_state_path.exists() and checkpoint_index_path.exists() and checkpoint_meta_path.exists():
            try:
                state = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))
                if (
                    state.get("source_size") == source_stat.st_size
                    and str(Path(state.get("source_file", "")).resolve()) == str(source_path.resolve())
                ):
                    self.index = faiss.read_index(str(checkpoint_index_path))
                    restored_docs = []
                    with checkpoint_meta_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            if line.strip():
                                restored_docs.append(json.loads(line))
                    self.documents = restored_docs
                    input_records = int(state.get("input_records", 0))
                    valid_indexed = int(state.get("valid_indexed", len(self.documents)))
                    skipped_records = int(state.get("skipped_records", 0))
                    resumed = True
                    print(
                        f"[{dataset_name or 'Index'}] Resuming from checkpoint: "
                        f"{valid_indexed:,} documents previously indexed (fast-forwarding {input_records:,} input records)..."
                    )
            except Exception as exc:
                print(f"[{dataset_name or 'Index'}] WARNING: Checkpoint recovery failed ({exc}); starting fresh.")
                self.index = None
                self.documents = []
                input_records = 0
                valid_indexed = 0
                skipped_records = 0
                resumed = False

        if not resumed:
            for chk in (checkpoint_index_path, checkpoint_meta_path, checkpoint_state_path):
                if chk.exists():
                    try:
                        chk.unlink()
                    except OSError:
                        pass

        resumed_count = valid_indexed
        start_time = time.perf_counter()
        current_chunk_docs: List[dict] = []
        current_chunk_texts: List[str] = []
        lines_to_skip = input_records if resumed else 0
        current_line_idx = 0

        def _flush_chunk():
            nonlocal current_chunk_docs, current_chunk_texts
            if not current_chunk_texts:
                return
            chunk_n = len(current_chunk_docs)
            t0 = time.perf_counter()
            vectors = embedding_model.encode_documents(
                current_chunk_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            )
            chunk_time = time.perf_counter() - t0
            if self.index is None:
                dim = vectors.shape[1]
                self.index = faiss.IndexFlatIP(dim)
            self.index.add(np.asarray(vectors, dtype=np.float32))
            self.documents.extend(current_chunk_docs)

            # Checkpoint index to disk
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_chk_idx = checkpoint_index_path.with_name(f"{checkpoint_index_path.stem}.tmp{checkpoint_index_path.suffix}")
            faiss.write_index(self.index, str(tmp_chk_idx))
            tmp_chk_idx.replace(checkpoint_index_path)

            # Checkpoint metadata to append-only JSONL
            with checkpoint_meta_path.open("a", encoding="utf-8") as meta_out:
                for doc in current_chunk_docs:
                    meta_out.write(json.dumps(doc, ensure_ascii=False) + "\n")

            # Checkpoint state
            state_payload = {
                "source_file": str(source_path.resolve()),
                "source_size": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "input_records": input_records,
                "valid_indexed": valid_indexed,
                "valid_indexed_documents": valid_indexed,
                "skipped_records": skipped_records,
                "dataset_name": dataset_name,
            }
            tmp_chk_state = checkpoint_state_path.with_name(f"{checkpoint_state_path.stem}.tmp{checkpoint_state_path.suffix}")
            tmp_chk_state.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
            tmp_chk_state.replace(checkpoint_state_path)

            total_elapsed = time.perf_counter() - start_time
            chunk_rate = chunk_n / max(chunk_time, 1e-6)
            overall_rate = (valid_indexed - resumed_count) / max(total_elapsed, 1e-6)

            print(
                f"[{dataset_name or 'Index'}] Chunk: +{chunk_n:,} docs in {chunk_time:.1f}s ({chunk_rate:.1f} docs/s) | "
                f"Total indexed: {valid_indexed:,} | Elapsed: {total_elapsed / 60:.1f}m | Overall rate: {overall_rate:.1f} docs/s"
            )

            current_chunk_docs = []
            current_chunk_texts = []
            if progress_callback:
                try:
                    progress_callback(valid_indexed, input_records)
                except TypeError:
                    progress_callback(valid_indexed)

        try:
            with source_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    current_line_idx += 1
                    if current_line_idx <= lines_to_skip:
                        continue
                    if not line.strip():
                        continue
                    input_records += 1
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        skipped_records += 1
                        continue

                    text = doc.get("text") or doc.get("content") or doc.get("passage")
                    if not text or not str(text).strip():
                        skipped_records += 1
                        continue

                    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
                    valid_doc = {
                        "doc_id": str(doc.get("doc_id") or doc.get("id") or doc.get("document_id") or valid_indexed),
                        "text": str(text),
                        "title": str(doc.get("title") or metadata.get("title") or ""),
                        "source": str(doc.get("source") or dataset_name or "unknown"),
                        "split": str(doc.get("split") or "train"),
                        "metadata": metadata,
                    }
                    for key in ("question_id", "conversation_id", "turn_id"):
                        if doc.get(key) is not None:
                            valid_doc[key] = doc[key]

                    current_chunk_docs.append(valid_doc)
                    current_chunk_texts.append(str(text))
                    valid_indexed += 1

                    if len(current_chunk_docs) >= chunk_size:
                        _flush_chunk()

                    if limit is not None and valid_indexed >= limit:
                        break

            if current_chunk_docs:
                _flush_chunk()

            if self.index is None or self.index.ntotal == 0:
                raise ValueError("No valid document records were found to build the FAISS index.")
            if self.index.ntotal != len(self.documents):
                raise ValueError(
                    f"FAISS vector count ({self.index.ntotal}) does not match document count ({len(self.documents)})"
                )

            # Save to staging artifacts first
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(staging_index_path))
            payload = [
                {**doc, "metadata": dict(doc.get("metadata", {})) if isinstance(doc.get("metadata", {}), dict) else {}}
                for doc in self.documents
            ]
            with staging_metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)

            # Atomically publish completed staging artifacts
            staging_index_path.replace(self.index_path)
            staging_metadata_path.replace(self.metadata_path)

            # Clean up completed checkpoint files
            for chk in (checkpoint_index_path, checkpoint_meta_path, checkpoint_state_path):
                if chk.exists():
                    try:
                        chk.unlink()
                    except OSError:
                        pass

        except Exception:
            # Ensure failed or aborted builds never leave corrupt staging files or overwrite existing completed index
            for stale in (staging_index_path, staging_metadata_path):
                if stale.exists():
                    try:
                        stale.unlink()
                    except OSError:
                        pass
            raise

        return {
            "dataset": dataset_name or self.index_path.stem,
            "input_records": input_records,
            "valid_indexed_documents": valid_indexed,
            "skipped_records": skipped_records,
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "limit": limit,
            "resumed": resumed,
        }


