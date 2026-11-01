"""
Frozen Snapshot of CARO RAG Pipeline from CARO_RAG.ipynb.
HARD RULE: Do not fix, alter, or 'improve' CARO logic.
Assets are loaded once and shared.
"""

import os
import sys
import time
import json
import re
import math
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Set
import urllib.request
import urllib.error
from contextvars import ContextVar

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Locate project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEXES_DIR = PROJECT_ROOT / "indexes" / "hotpotqa"

FAISS_INDEX_PATH = INDEXES_DIR / "hotpotqa_full_index.faiss"
METADATA_PATH = INDEXES_DIR / "hotpotqa_full_index.meta.json"
BM25_CACHE_PATH = INDEXES_DIR / "bm25_cache.npz"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL_NAME = "gemma3:4b"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

EXPECTED_DOC_COUNT = 250000
EXPECTED_VECTOR_DIM = 1024

INITIAL_TOP_K = 30


def print_ram_usage(context: str = "") -> None:
    """Prints current process RSS and system RAM percentage using psutil."""
    try:
        import psutil
        proc = psutil.Process()
        proc_mb = proc.memory_info().rss / (1024 * 1024)
        vm = psutil.virtual_memory()
        used_gb = vm.used / (1024 * 1024 * 1024)
        total_gb = vm.total / (1024 * 1024 * 1024)
        prefix = f"[RAM: {context}]" if context else "[RAM]"
        print(f"{prefix} Process RSS: {proc_mb:.1f} MB | System RAM: {used_gb:.2f} / {total_gb:.2f} GB ({vm.percent}%)", flush=True)
    except Exception as e:
        print(f"[RAM] Warning: Could not read memory stats: {e}", flush=True)
EXPANDED_TOP_K = 15
MAX_QUERY_VARIANTS = 6
MAX_CONTEXT_DOCS = 8
UNANSWERABLE_STRING = "The relevant document is not present in the corpus."

# Global context variable for document exclusion (thread-safe, shared across retrievers)
_active_exclusion_var: ContextVar[Optional[Set[str]]] = ContextVar("active_exclusion", default=None)

# Global call counters (ContextVar-enabled for instrumentation)
_call_tracker_var: ContextVar[Optional[Dict[str, int]]] = ContextVar("call_tracker", default=None)


class CallTracker:
    """Context manager to track LLM and retrieval calls per query."""
    def __init__(self):
        self.stats = {"llm_calls": 0, "retrieval_calls": 0}
        self._token = None

    def __enter__(self):
        self._token = _call_tracker_var.set(self.stats)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token:
            _call_tracker_var.reset(self._token)

    @property
    def llm_calls(self) -> int:
        return self.stats["llm_calls"]

    @property
    def retrieval_calls(self) -> int:
        return self.stats["retrieval_calls"]


def record_llm_call():
    tracker = _call_tracker_var.get()
    if tracker is not None:
        tracker["llm_calls"] += 1


def record_retrieval_call():
    tracker = _call_tracker_var.get()
    if tracker is not None:
        tracker["retrieval_calls"] += 1


class ExclusionContext:
    """Context manager to enforce document exclusion across all retrievers."""
    def __init__(self, excluded_doc_ids: Optional[Set[str]]):
        self.excluded = set(excluded_doc_ids) if excluded_doc_ids else None
        self._token = None

    def __enter__(self):
        self._token = _active_exclusion_var.set(self.excluded)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token:
            _active_exclusion_var.reset(self._token)


class CSRPostingsWrapper:
    """Memory-efficient CSR postings wrapper exposing dict-like .get(token, [])."""
    def __init__(self, offsets: np.ndarray, doc_ids: np.ndarray, tfs: np.ndarray, vocab_to_id: Dict[str, int]):
        self.offsets = offsets
        self.doc_ids = doc_ids
        self.tfs = tfs
        self.vocab_to_id = vocab_to_id

    def get(self, token: str, default=None):
        tid = self.vocab_to_id.get(token)
        if tid is None:
            return default if default is not None else []
        s, e = self.offsets[tid], self.offsets[tid + 1]
        return list(zip(self.doc_ids[s:e].tolist(), self.tfs[s:e].tolist()))

    def __contains__(self, token: str) -> bool:
        return token in self.vocab_to_id

    def __len__(self) -> int:
        return len(self.vocab_to_id)


class CSRTitlePostingsWrapper:
    """Memory-efficient CSR title postings wrapper exposing dict-like .get(token, [])."""
    def __init__(self, offsets: np.ndarray, doc_ids: np.ndarray, vocab_to_id: Dict[str, int]):
        self.offsets = offsets
        self.doc_ids = doc_ids
        self.vocab_to_id = vocab_to_id

    def get(self, token: str, default=None):
        tid = self.vocab_to_id.get(token)
        if tid is None:
            return default if default is not None else []
        s, e = self.offsets[tid], self.offsets[tid + 1]
        return self.doc_ids[s:e].tolist()

    def __contains__(self, token: str) -> bool:
        return token in self.vocab_to_id

    def __len__(self) -> int:
        return len(self.vocab_to_id)


# Asset cache singleton
class CorpusAssets:
    faiss_index: Optional[faiss.Index] = None
    corpus_documents: Optional[List[Dict[str, Any]]] = None
    doc_id_to_idx: Optional[Dict[str, int]] = None
    title_to_doc_ids: Optional[Dict[str, List[str]]] = None
    embedding_model: Optional[SentenceTransformer] = None

    body_offsets: Optional[np.ndarray] = None
    body_postings_doc_ids: Optional[np.ndarray] = None
    body_postings_tfs: Optional[np.ndarray] = None
    body_vocab_to_id: Optional[Dict[str, int]] = None

    title_offsets: Optional[np.ndarray] = None
    title_postings_doc_ids: Optional[np.ndarray] = None
    title_vocab_to_id: Optional[Dict[str, int]] = None

    sparse_title_index: Optional[Any] = None
    sparse_inv_index: Optional[Any] = None
    corpus_doc_lengths: Optional[np.ndarray] = None
    avg_doc_length: float = 0.0
    total_docs: int = 0
    _initialized: bool = False

    @classmethod
    def load(cls, device: Optional[str] = None):
        target_device = device or os.environ.get("BGE_M3_DEVICE", "cpu")
        if target_device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[CorpusAssets] Warning: CUDA requested but not available. Falling back to CPU.", flush=True)
                    target_device = "cpu"
            except ImportError:
                target_device = "cpu"

        if cls._initialized:
            if cls.embedding_model is not None:
                try:
                    curr_dev = str(cls.embedding_model.device).split(":")[0]
                    if curr_dev != target_device:
                        cls.embedding_model = cls.embedding_model.to(target_device)
                        print(f"[CorpusAssets] Moved embedding model to {target_device}.", flush=True)
                except Exception as e:
                    print(f"[CorpusAssets] Warning: Failed to move embedding model: {e}", flush=True)
            return

        print(f"[CorpusAssets] Loading FAISS index from {FAISS_INDEX_PATH} (mmap mode)...")
        t0 = time.time()
        cls.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH), faiss.IO_FLAG_MMAP)
        print_ram_usage("FAISS index loaded (mmap)")

        clean_cache_path = FAISS_INDEX_PATH.parent / "clean_docs_cache.pkl"
        if clean_cache_path.exists():
            print(f"[CorpusAssets] Loading cleaned metadata from cache {clean_cache_path}...")
            import pickle
            with open(clean_cache_path, "rb") as f:
                cached_data = pickle.load(f)
            cls.corpus_documents = cached_data["corpus_documents"]
            cls.doc_id_to_idx = cached_data["doc_id_to_idx"]
            cls.title_to_doc_ids = cached_data["title_to_doc_ids"]
            del cached_data
        else:
            print(f"[CorpusAssets] Loading metadata from {METADATA_PATH}...")
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                corpus_documents = json.load(f)
            if isinstance(corpus_documents, dict):
                for _key in ("documents", "data", "items", "metadata"):
                    if isinstance(corpus_documents.get(_key), list):
                        corpus_documents = corpus_documents[_key]
                        break

            clean_corpus_documents = []
            doc_id_to_idx = {}
            title_to_doc_ids = defaultdict(list)
            for idx, doc in enumerate(corpus_documents):
                did = str(doc.get("doc_id") or doc.get("id") or idx)
                text = str(doc.get("text") or doc.get("content") or "")
                title = str(doc.get("title") or "").strip()
                clean_corpus_documents.append({"doc_id": did, "title": title, "text": text})
                doc_id_to_idx[did] = idx
                if title:
                    title_to_doc_ids[title].append(did)

            del corpus_documents
            cls.corpus_documents = clean_corpus_documents
            cls.doc_id_to_idx = doc_id_to_idx
            cls.title_to_doc_ids = dict(title_to_doc_ids)

            import pickle
            try:
                with open(clean_cache_path, "wb") as f:
                    pickle.dump({
                        "corpus_documents": cls.corpus_documents,
                        "doc_id_to_idx": cls.doc_id_to_idx,
                        "title_to_doc_ids": cls.title_to_doc_ids
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[CorpusAssets] Saved clean metadata cache to {clean_cache_path}.")
            except Exception as e:
                print(f"[CorpusAssets] Warning: Failed to save pickle cache: {e}")

        import gc
        gc.collect()

        if BM25_CACHE_PATH.exists():
            print(f"[CorpusAssets] Loading BM25 index from cache {BM25_CACHE_PATH}...")
            cache = np.load(BM25_CACHE_PATH, allow_pickle=True)
            cls.body_offsets = np.asarray(cache["body_offsets"], dtype=np.int64)
            cls.body_postings_doc_ids = np.asarray(cache["body_postings_doc_ids"], dtype=np.int32)
            cls.body_postings_tfs = np.asarray(cache["body_postings_tfs"], dtype=np.int16)
            body_vocab = [str(x) for x in cache["body_vocab"]]
            cls.body_vocab_to_id = {w: i for i, w in enumerate(body_vocab)}

            cls.title_offsets = np.asarray(cache["title_offsets"], dtype=np.int64)
            cls.title_postings_doc_ids = np.asarray(cache["title_postings_doc_ids"], dtype=np.int32)
            title_vocab = [str(x) for x in cache["title_vocab"]]
            cls.title_vocab_to_id = {w: i for i, w in enumerate(title_vocab)}

            cls.corpus_doc_lengths = np.asarray(cache["corpus_doc_lengths"], dtype=np.int32)
            cls.avg_doc_length = float(cache["avg_doc_length"])
            cls.total_docs = int(cache["total_docs"])

            cls.sparse_inv_index = CSRPostingsWrapper(cls.body_offsets, cls.body_postings_doc_ids, cls.body_postings_tfs, cls.body_vocab_to_id)
            cls.sparse_title_index = CSRTitlePostingsWrapper(cls.title_offsets, cls.title_postings_doc_ids, cls.title_vocab_to_id)
            print_ram_usage("BM25 index loaded")
        else:
            print(f"[CorpusAssets] Building memory-efficient BM25 index and saving to {BM25_CACHE_PATH}...")
            _word_regex = re.compile(r"[a-z0-9]+")
            _stopwords = {
                'when', 'was', 'in', 'is', 'a', 'an', 'the', 'and', 'or', 'to', 'of', 'for', 'on', 'at',
                'by', 'from', 'with', 'about', 'who', 'what', 'where', 'why', 'how', 'did', 'its', 'it',
                'were', 'been', 'has', 'have', 'had', 'that', 'this', 'these', 'those', 'which', 'does',
                'do', 'are', 'is', 'as', 'be', 'their', 'they', 'he', 'she', 'his', 'her'
            }

            body_df = Counter()
            title_df = Counter()
            corpus_doc_lengths = np.empty(len(clean_corpus_documents), dtype=np.int32)

            for idx, doc in enumerate(clean_corpus_documents):
                t_toks = [w for w in _word_regex.findall(doc["title"].lower()) if w not in _stopwords]
                b_toks = [w for w in _word_regex.findall(doc["text"].lower()) if w not in _stopwords]
                corpus_doc_lengths[idx] = len(b_toks) + 2 * len(t_toks)
                for w in set(b_toks):
                    body_df[w] += 1
                for w in set(t_toks):
                    title_df[w] += 1

            body_vocab = sorted(body_df.keys())
            cls.body_vocab_to_id = {w: i for i, w in enumerate(body_vocab)}
            cls.body_offsets = np.zeros(len(body_vocab) + 1, dtype=np.int64)
            np.cumsum([body_df[w] for w in body_vocab], out=cls.body_offsets[1:])
            total_body_postings = int(cls.body_offsets[-1])
            cls.body_postings_doc_ids = np.empty(total_body_postings, dtype=np.int32)
            cls.body_postings_tfs = np.empty(total_body_postings, dtype=np.int16)
            body_write_ptr = cls.body_offsets[:-1].copy()

            title_vocab = sorted(title_df.keys())
            cls.title_vocab_to_id = {w: i for i, w in enumerate(title_vocab)}
            cls.title_offsets = np.zeros(len(title_vocab) + 1, dtype=np.int64)
            np.cumsum([title_df[w] for w in title_vocab], out=cls.title_offsets[1:])
            total_title_postings = int(cls.title_offsets[-1])
            cls.title_postings_doc_ids = np.empty(total_title_postings, dtype=np.int32)
            title_write_ptr = cls.title_offsets[:-1].copy()

            for idx, doc in enumerate(clean_corpus_documents):
                t_toks = [w for w in _word_regex.findall(doc["title"].lower()) if w not in _stopwords]
                b_toks = [w for w in _word_regex.findall(doc["text"].lower()) if w not in _stopwords]
                for w, tf in Counter(b_toks).items():
                    tid = cls.body_vocab_to_id[w]
                    pos = body_write_ptr[tid]
                    cls.body_postings_doc_ids[pos] = idx
                    cls.body_postings_tfs[pos] = min(tf, 32767)
                    body_write_ptr[tid] += 1
                for w in set(t_toks):
                    tid = cls.title_vocab_to_id[w]
                    pos = title_write_ptr[tid]
                    cls.title_postings_doc_ids[pos] = idx
                    title_write_ptr[tid] += 1

            cls.corpus_doc_lengths = corpus_doc_lengths
            cls.avg_doc_length = float(corpus_doc_lengths.sum() / max(len(clean_corpus_documents), 1))
            cls.total_docs = len(clean_corpus_documents)

            staging_path = BM25_CACHE_PATH.with_name("bm25_cache.staging.npz")
            BM25_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                staging_path,
                body_offsets=cls.body_offsets,
                body_postings_doc_ids=cls.body_postings_doc_ids,
                body_postings_tfs=cls.body_postings_tfs,
                body_vocab=np.array(body_vocab, dtype=str),
                title_offsets=cls.title_offsets,
                title_postings_doc_ids=cls.title_postings_doc_ids,
                title_vocab=np.array(title_vocab, dtype=str),
                corpus_doc_lengths=cls.corpus_doc_lengths,
                avg_doc_length=np.float64(cls.avg_doc_length),
                total_docs=np.int32(cls.total_docs),
            )
            staging_path.replace(BM25_CACHE_PATH)

            cls.sparse_inv_index = CSRPostingsWrapper(cls.body_offsets, cls.body_postings_doc_ids, cls.body_postings_tfs, cls.body_vocab_to_id)
            cls.sparse_title_index = CSRTitlePostingsWrapper(cls.title_offsets, cls.title_postings_doc_ids, cls.title_vocab_to_id)

        import gc
        gc.collect()

        print(f"[CorpusAssets] Loading embedding model {EMBEDDING_MODEL_NAME} on {target_device}...")
        cls.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=target_device)
        print_ram_usage(f"BGE-M3 model loaded ({target_device})")

        cls._initialized = True
        print(f"[CorpusAssets] Ready in {time.time() - t0:.2f}s | Documents: {cls.total_docs}")
        print_ram_usage("CorpusAssets ready")



def estimate_tokens(text: str) -> int:
    """Approximate token count for Gemma (approx 3.5 chars/token or 1.3 * words)."""
    if not text:
        return 0
    return max(int(len(text) / 3.5), int(len(text.split()) * 1.3))


def enforce_prompt_token_guard(prompt: str, max_tokens: int = 1800) -> str:
    """
    Guard: if prompt exceeds ~1800 tokens, reduce document length or drop documents and log it,
    so Ollama (configured with num_ctx=2048) never truncates silently.
    """
    est = estimate_tokens(prompt)
    if est <= max_tokens:
        return prompt

    orig_tokens = est
    # 1. Progressively trim document lines (e.g. Text: or Content:)
    lines = prompt.splitlines(keepends=True)
    for target_line_len in (400, 250, 150):
        trimmed_lines = []
        for line in lines:
            if any(k in line for k in ("Text: ", "Content: ")) and len(line) > target_line_len:
                trimmed_lines.append(line[:target_line_len].rstrip() + "...\n")
            else:
                trimmed_lines.append(line)
        candidate = "".join(trimmed_lines)
        if estimate_tokens(candidate) <= max_tokens:
            new_tokens = estimate_tokens(candidate)
            print(f"[call_gemma Guard] Prompt exceeded ~1800 tokens ({orig_tokens} est). Reduced doc text lines to {target_line_len} chars -> {new_tokens} est tokens.", flush=True)
            return candidate

    # 2. If still exceeding, drop the last document
    doc_pattern = re.compile(r'(Document \[\d+\]|\[\d+\] ID:)', re.IGNORECASE)
    matches = list(doc_pattern.finditer(prompt))
    if len(matches) > 1:
        keep_until = matches[-1].start()
        # Find closing instruction if present
        closing = ""
        if "Write your answer" in prompt:
            closing = "\n\nWrite your answer following the exact format requirements above."
        elif "Return JSON only:" in prompt:
            closing = "\n\nReturn JSON only:"
        candidate = prompt[:keep_until].rstrip() + closing
        new_tokens = estimate_tokens(candidate)
        print(f"[call_gemma Guard] Prompt exceeded ~1800 tokens ({orig_tokens} est). Dropped last document -> {new_tokens} est tokens.", flush=True)
        return candidate

    # 3. Direct truncation fallback
    char_limit = int(max_tokens * 3.4)
    candidate = prompt[:char_limit] + "\n[... truncated by guard to fit context window ...]\n"
    new_tokens = estimate_tokens(candidate)
    print(f"[call_gemma Guard] Prompt exceeded ~1800 tokens ({orig_tokens} est). Truncated prompt to fit context window -> {new_tokens} est tokens.", flush=True)
    return candidate


def call_gemma(
    prompt: str,
    json_mode: bool = False,
    timeout: int = 300,
    max_retries: int = 2,
    num_predict: Optional[int] = None
) -> Dict[str, Any]:
    """Safe, robust caller for Gemma 3 4B via Ollama HTTP API with instrumented call counting and context guard."""
    prompt = enforce_prompt_token_guard(prompt, max_tokens=1800)
    record_llm_call()
    url = f"{OLLAMA_BASE_URL}/api/generate"
    options: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 0.9,
        "num_ctx": 2048,
    }
    if num_predict is not None:
        options["num_predict"] = int(num_predict)

    payload = {
        "model": LLM_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": options
    }
    if json_mode:
        payload["format"] = "json"

    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )

    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                raw_text = res_json.get("response", "").strip()

                if json_mode:
                    try:
                        parsed = json.loads(raw_text)
                        return {"success": True, "raw": raw_text, "json": parsed, "error": None}
                    except json.JSONDecodeError:
                        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                        if match:
                            try:
                                parsed = json.loads(match.group(0))
                                return {"success": True, "raw": raw_text, "json": parsed, "error": None}
                            except Exception as parse_err:
                                last_error = f"Malformed JSON: {parse_err}"
                        else:
                            last_error = f"No JSON object found in response: {raw_text[:100]}"
                else:
                    return {"success": True, "raw": raw_text, "json": None, "error": None}
        except urllib.error.URLError as e:
            last_error = f"Connection error to Ollama at {url}: {e.reason}"
        except Exception as e:
            last_error = f"Request error: {str(e)}"
        time.sleep(0.5)

    return {"success": False, "raw": "", "json": None, "error": last_error}


@dataclass
class RoutingDecision:
    route: str  # "DIRECT", "MEMORY", "RETRIEVE"
    question_type: str  # "chitchat", "factual", "multi_hop", "follow_up"
    requires_memory: bool
    requires_retrieval: bool
    query_rewritten: str
    reason: str = ""


def route_query_with_llm(query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> RoutingDecision:
    history_str = "None"
    if conversation_history and len(conversation_history) > 0:
        history_lines = []
        for turn in conversation_history[-4:]:
            history_lines.append(f"{turn['role'].upper()}: {turn['content']}")
        history_str = "\n".join(history_lines)

    prompt = f"""Route query for 250k-doc RAG system.
Categories:
- "DIRECT": Greetings, general chat, identity, math, coding; no external facts needed.
- "MEMORY": Answerable from conversation history alone; resolves prior pronouns.
- "RETRIEVE": Real-world facts, entities, dates requiring corpus search.

History: {history_str}
Query: {query}

JSON only:
{{"route":"DIRECT|MEMORY|RETRIEVE","type":"chitchat|factual|multi_hop|follow_up","mem":true|false,"ret":true|false,"query":"<standalone query>"}}"""

    res = call_gemma(prompt, json_mode=True, timeout=40, num_predict=120)
    if res["success"] and res["json"]:
        parsed = res["json"]
        route = str(parsed.get("route", "")).strip().upper()
        if route not in ("DIRECT", "MEMORY", "RETRIEVE"):
            if parsed.get("ret", parsed.get("requires_retrieval", False)):
                route = "RETRIEVE"
            elif parsed.get("mem", parsed.get("requires_memory", False)):
                route = "MEMORY"
            else:
                route = "DIRECT"

        q_type = str(parsed.get("type", parsed.get("question_type", "factual"))).strip().lower()
        req_mem = bool(parsed.get("mem", parsed.get("requires_memory", False)))
        req_ret = bool(parsed.get("ret", parsed.get("requires_retrieval", True if route == "RETRIEVE" else False)))
        reason = str(parsed.get("reason", ""))
        rewritten = str(parsed.get("query", parsed.get("query_rewritten", query))).strip() or query

        return RoutingDecision(
            route=route,
            question_type=q_type,
            requires_memory=req_mem,
            requires_retrieval=req_ret,
            query_rewritten=rewritten,
            reason=reason
        )
    else:
        return RoutingDecision(
            route="RETRIEVE",
            question_type="factual",
            requires_memory=bool(conversation_history and len(conversation_history) > 0),
            requires_retrieval=True,
            reason=f"LLM routing fallback triggered ({res.get('error') or 'malformed output'}). Safe default: RETRIEVE.",
            query_rewritten=query
        )


def select_retrieval_strategy(routing: RoutingDecision) -> Dict[str, Any]:
    if routing.route == "DIRECT":
        return {"strategy": "DIRECT", "confidence": 0.95, "reason": "Direct response; no corpus retrieval needed."}
    if routing.route == "MEMORY" and not routing.requires_retrieval:
        return {"strategy": "MEMORY", "confidence": 0.90, "reason": "Resolved purely from prior conversational context."}
    if routing.requires_memory and routing.requires_retrieval:
        return {"strategy": "MEMORY_AND_RETRIEVAL", "confidence": 0.92, "reason": "Requires both conversation memory resolution and corpus retrieval."}

    q_lower = routing.query_rewritten.lower()
    is_multi_hop = routing.question_type == "multi_hop" or "both" in q_lower or "and" in q_lower
    keyword_pattern = r"\b(population|isbn|postcode|zip code|release date|official name|born|birthplace|located|founded)\b"
    has_keyword_terms = bool(re.search(keyword_pattern, q_lower))
    has_named_entity = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", routing.query_rewritten))

    if is_multi_hop:
        return {"strategy": "HYBRID", "confidence": 0.90, "reason": "Multi-hop query benefits from combining semantic dense retrieval with exact sparse matching."}
    elif has_keyword_terms and has_named_entity:
        return {"strategy": "HYBRID", "confidence": 0.90, "reason": "Specific keyword terms and named entity present; combining dense semantic search with exact lexical match via HYBRID."}
    else:
        return {"strategy": "DENSE", "confidence": 0.88, "reason": "Semantic similarity over dense vector space is well-suited for general conceptual query."}


def _doc_text(doc: Dict[str, Any]) -> str:
    return str(doc.get("text", doc.get("content", "")))


def search_dense(query: str, top_k: int = 30, excluded_doc_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """Dense semantic retrieval with over-fetching and exclusion filtering."""
    record_retrieval_call()
    if not str(query).strip():
        return []

    active_exclusion = excluded_doc_ids or _active_exclusion_var.get() or set()
    buffer = len(active_exclusion) + 50 if active_exclusion else 0
    fetch_k = min(max(1, int(top_k) + buffer), CorpusAssets.faiss_index.ntotal)

    query_vec = CorpusAssets.embedding_model.encode(query, normalize_embeddings=True)
    query_arr = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    if query_arr.shape[1] != CorpusAssets.faiss_index.d:
        raise ValueError(f"Query embedding dimension {query_arr.shape[1]} != index dimension {CorpusAssets.faiss_index.d}")

    scores, indices = CorpusAssets.faiss_index.search(query_arr, fetch_k)
    results = []
    rank = 1
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(CorpusAssets.corpus_documents):
            continue
        doc = dict(CorpusAssets.corpus_documents[int(idx)])
        did = str(doc.get("doc_id", doc.get("id", "")))
        if active_exclusion and did in active_exclusion:
            continue
        doc.update({"retrieval_score": float(score), "retrieval_rank": rank, "retrieval_type": "dense"})
        results.append(doc)
        rank += 1
        if len(results) >= top_k:
            break
    return results


def search_sparse(query: str, top_k: int = 30, excluded_doc_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """BM25 retrieval with term frequency, document length normalization, and title boost."""
    record_retrieval_call()
    _word_regex = re.compile(r"[a-z0-9]+")
    _stopwords = {
        'when', 'was', 'in', 'is', 'a', 'an', 'the', 'and', 'or', 'to', 'of', 'for', 'on', 'at',
        'by', 'from', 'with', 'about', 'who', 'what', 'where', 'why', 'how', 'did', 'its', 'it',
        'were', 'been', 'has', 'have', 'had', 'that', 'this', 'these', 'those', 'which', 'does',
        'do', 'are', 'is', 'as', 'be', 'their', 'they', 'he', 'she', 'his', 'her'
    }
    raw_tokens = _word_regex.findall(str(query).lower())
    q_tokens = [w for w in raw_tokens if w not in _stopwords] or raw_tokens
    if not q_tokens:
        return []

    active_exclusion = excluded_doc_ids or _active_exclusion_var.get() or set()
    buffer = len(active_exclusion) + 50 if active_exclusion else 0
    fetch_k = max(1, int(top_k) + buffer)

    k1, b = 1.5, 0.75
    doc_scores: Dict[int, float] = defaultdict(float)
    total_docs = CorpusAssets.total_docs
    avg_doc_length = CorpusAssets.avg_doc_length

    for token in set(q_tokens):
        if CorpusAssets.body_vocab_to_id is not None and token in CorpusAssets.body_vocab_to_id:
            tid = CorpusAssets.body_vocab_to_id[token]
            start = CorpusAssets.body_offsets[tid]
            end = CorpusAssets.body_offsets[tid + 1]
            df = end - start
            if df:
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                docs = CorpusAssets.body_postings_doc_ids[start:end]
                tfs = CorpusAssets.body_postings_tfs[start:end]
                dl = CorpusAssets.corpus_doc_lengths[docs]
                denom = tfs + k1 * (1.0 - b + b * (dl / max(avg_doc_length, 1.0)))
                term_scores = idf * (tfs * (k1 + 1.0)) / denom
                for doc_idx, s in zip(docs, term_scores):
                    doc_scores[int(doc_idx)] += float(s)
        elif CorpusAssets.sparse_inv_index is not None:
            body_postings = CorpusAssets.sparse_inv_index.get(token, [])
            df = len(body_postings)
            if df:
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                for doc_idx, tf in body_postings:
                    dl = CorpusAssets.corpus_doc_lengths[doc_idx]
                    denom = tf + k1 * (1.0 - b + b * (dl / max(avg_doc_length, 1.0)))
                    doc_scores[doc_idx] += idf * (tf * (k1 + 1.0)) / denom

        if CorpusAssets.title_vocab_to_id is not None and token in CorpusAssets.title_vocab_to_id:
            tid = CorpusAssets.title_vocab_to_id[token]
            start = CorpusAssets.title_offsets[tid]
            end = CorpusAssets.title_offsets[tid + 1]
            df = end - start
            if df:
                title_idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                boost = 2.0 * title_idf
                for doc_idx in CorpusAssets.title_postings_doc_ids[start:end]:
                    doc_scores[int(doc_idx)] += boost
        elif CorpusAssets.sparse_title_index is not None:
            title_postings = CorpusAssets.sparse_title_index.get(token, [])
            if title_postings:
                title_idf = math.log(1.0 + (total_docs - len(title_postings) + 0.5) / (len(title_postings) + 0.5))
                for doc_idx in title_postings:
                    doc_scores[doc_idx] += 2.0 * title_idf

    if not doc_scores:
        return []

    sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:fetch_k]
    max_score = sorted_items[0][1] if sorted_items else 1.0

    results = []
    rank = 1
    for doc_idx, raw_score in sorted_items:
        doc = dict(CorpusAssets.corpus_documents[doc_idx])
        did = str(doc.get("doc_id", doc.get("id", "")))
        if active_exclusion and did in active_exclusion:
            continue
        doc.update({
            "retrieval_score": float(raw_score / max(max_score, 1e-9)),
            "raw_sparse_score": float(raw_score),
            "retrieval_rank": rank,
            "retrieval_type": "sparse"
        })
        results.append(doc)
        rank += 1
        if len(results) >= top_k:
            break
    return results


def search_hybrid(query: str, top_k: int = 30, excluded_doc_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """Hybrid retrieval via Reciprocal Rank Fusion (RRF), preserving strategy semantics."""
    active_exclusion = excluded_doc_ids or _active_exclusion_var.get() or set()
    buffer = len(active_exclusion) + 50 if active_exclusion else 0
    fetch_k = max(top_k * 2, top_k) + buffer

    dense_res = search_dense(query, top_k=fetch_k, excluded_doc_ids=active_exclusion)
    sparse_res = search_sparse(query, top_k=fetch_k, excluded_doc_ids=active_exclusion)

    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Dict[str, Any]] = {}
    for results in (dense_res, sparse_res):
        for d in results:
            did = str(d.get("doc_id", d.get("id", "")))
            if not did or (active_exclusion and did in active_exclusion):
                continue
            doc_map.setdefault(did, d)
            rrf_scores[did] += 1.0 / (60 + int(d.get("retrieval_rank", 9999)))

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:max(1, int(top_k))]
    results = []
    max_score = ranked[0][1] if ranked else 1.0
    for rank, (did, score) in enumerate(ranked, start=1):
        doc = dict(doc_map[did])
        doc.update({
            "retrieval_score": float(score / max(max_score, 1e-9)),
            "rrf_score": float(score),
            "retrieval_rank": rank,
            "retrieval_type": "hybrid"
        })
        results.append(doc)
    return results


def expand_query(query: str) -> List[str]:
    prompt = f"""Create at most 3 distinct, short search queries that could retrieve documents answering this question.
Preserve named entities and dates. Do not answer the question.
Question: {query}
Return only a JSON array of strings."""
    res = call_gemma(prompt, json_mode=True, timeout=30)
    candidates = res["json"] if res["success"] and isinstance(res["json"], list) else []
    cleaned = []
    for item in candidates:
        item = str(item).strip()
        if item and item.casefold() != query.casefold() and item.casefold() not in {x.casefold() for x in cleaned}:
            cleaned.append(item)
    fallbacks = [f"{query} facts", f"{query} details", f"{query} official information"]
    for item in fallbacks:
        if len(cleaned) >= 3:
            break
        if item.casefold() not in {x.casefold() for x in cleaned}:
            cleaned.append(item)
    return cleaned[:3]


def decompose_query(query: str) -> List[str]:
    prompt = f"""Break this question into 2 or 3 independently searchable atomic questions.
If it is a single-fact question, return an empty array.
Do not answer the questions.
Question: {query}
Return only a JSON array of strings."""
    res = call_gemma(prompt, json_mode=True, timeout=30)
    candidates = res["json"] if res["success"] and isinstance(res["json"], list) else []
    return list(dict.fromkeys(str(x).strip() for x in candidates if str(x).strip() and str(x).strip().casefold() != query.casefold()))[:3]


def _merge_candidates(pass_results: List[Tuple[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    active_exclusion = _active_exclusion_var.get() or set()
    fused: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Dict[str, Any]] = {}
    for query_label, docs in pass_results:
        is_primary = "primary" in query_label.lower()
        pass_weight = 2.5 if is_primary else 1.0
        for rank, doc in enumerate(docs, start=1):
            did = str(doc.get("doc_id", doc.get("id", "")))
            if not did or (active_exclusion and did in active_exclusion):
                continue
            fused[did] += pass_weight / (60.0 + rank)
            if did not in doc_map:
                doc_map[did] = dict(doc)
            doc_map[did].setdefault("matched_queries", []).append(query_label)
    ranked_ids = sorted(fused, key=fused.get, reverse=True)
    output = []
    max_fused = fused[ranked_ids[0]] if ranked_ids else 1.0
    seen_titles: Set[str] = set()
    for did in ranked_ids:
        doc = dict(doc_map[did])
        raw_title = str(doc.get("title", "")).strip()
        title_key = raw_title.casefold()
        if title_key and title_key in seen_titles:
            continue
        if title_key:
            seen_titles.add(title_key)
        rank = len(output) + 1
        doc["fusion_score"] = float(fused[did])
        doc["retrieval_score"] = float(fused[did] / max(max_fused, 1e-9))
        doc["retrieval_rank"] = rank
        doc["retrieval_type"] = "multi_query_fusion"
        output.append(doc)
    return output


def execute_full_corpus_search(
    query: str,
    initial_strategy: str = "DENSE",
    min_confidence_score: float = 0.58,
    top_k: int = 8
) -> Dict[str, Any]:
    strategy_fns = {"DENSE": search_dense, "SPARSE": search_sparse, "HYBRID": search_hybrid}
    primary_strategy = initial_strategy if initial_strategy in strategy_fns else "DENSE"
    variants: List[str] = [str(query).strip()]
    log: List[Dict[str, Any]] = []
    pass_results: List[Tuple[str, List[Dict[str, Any]]]] = []
    search_timings: Dict[str, float] = {
        "retrieval_primary": 0.0,
        "retrieval_primary.query_encode": 0.0,
        "retrieval_primary.faiss_search": 0.0,
        "retrieval_primary.bm25_search": 0.0,
        "retrieval_primary.exclusion_filter": 0.0,
        "expansion": 0.0,
        "decomposition": 0.0,
        "retrieval_variants": 0.0,
        "retrieval_cross_strategy": 0.0,
        "retrieval_fusion": 0.0,
    }

    def run_pass(label: str, q: str, strategy: str, k: int):
        fn = strategy_fns[strategy]
        docs = fn(q, top_k=k)
        pass_results.append((f"{label} [{strategy}]", docs))
        log.append({"pass": len(log) + 1, "type": label, "strategy": strategy,
                    "query": q, "documents_returned": len(docs)})
        return docs

    # --- Instrumented primary pass with sub-timings ---
    t0 = time.time()
    active_exclusion = _active_exclusion_var.get() or set()
    buffer = len(active_exclusion) + 50 if active_exclusion else 0
    primary_k = max(INITIAL_TOP_K, top_k)
    fetch_k = min(max(1, primary_k + buffer), CorpusAssets.faiss_index.ntotal)

    # Sub-timing: query encode
    t_enc0 = time.time()
    if primary_strategy in ("DENSE", "HYBRID"):
        query_vec = CorpusAssets.embedding_model.encode(variants[0], normalize_embeddings=True)
        query_arr = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    else:
        query_arr = None
    search_timings["retrieval_primary.query_encode"] = round(time.time() - t_enc0, 4)

    # Sub-timing: FAISS search
    t_faiss0 = time.time()
    if query_arr is not None:
        faiss_scores, faiss_indices = CorpusAssets.faiss_index.search(query_arr, fetch_k)
    else:
        faiss_scores, faiss_indices = None, None
    search_timings["retrieval_primary.faiss_search"] = round(time.time() - t_faiss0, 4)

    # Sub-timing: BM25 search
    t_bm25_0 = time.time()
    if primary_strategy in ("SPARSE", "HYBRID"):
        _bm25_results = search_sparse(variants[0], top_k=primary_k + buffer)
    else:
        _bm25_results = []
    search_timings["retrieval_primary.bm25_search"] = round(time.time() - t_bm25_0, 4)

    # Sub-timing: exclusion filtering
    t_excl0 = time.time()
    if primary_strategy == "DENSE":
        record_retrieval_call()
        primary_docs = []
        rank = 1
        seen_primary_titles = set()
        for score, idx in zip(faiss_scores[0], faiss_indices[0]):
            if idx < 0 or idx >= len(CorpusAssets.corpus_documents):
                continue
            doc = dict(CorpusAssets.corpus_documents[int(idx)])
            did = str(doc.get("doc_id", doc.get("id", "")))
            if active_exclusion and did in active_exclusion:
                continue
            title_key = str(doc.get("title", "")).strip().casefold()
            if title_key and title_key in seen_primary_titles:
                continue
            if title_key:
                seen_primary_titles.add(title_key)
            doc.update({"retrieval_score": float(score), "retrieval_rank": rank, "retrieval_type": "dense"})
            primary_docs.append(doc)
            rank += 1
            if len(primary_docs) >= primary_k:
                break
    elif primary_strategy == "SPARSE":
        primary_docs = []
        seen_primary_titles = set()
        for d in _bm25_results:
            did = str(d.get("doc_id", d.get("id", "")))
            if active_exclusion and did in active_exclusion:
                continue
            title_key = str(d.get("title", "")).strip().casefold()
            if title_key and title_key in seen_primary_titles:
                continue
            if title_key:
                seen_primary_titles.add(title_key)
            primary_docs.append(d)
            if len(primary_docs) >= primary_k:
                break
    else:  # HYBRID
        dense_docs = []
        rank = 1
        for score, idx in zip(faiss_scores[0], faiss_indices[0]):
            if idx < 0 or idx >= len(CorpusAssets.corpus_documents):
                continue
            doc = dict(CorpusAssets.corpus_documents[int(idx)])
            did = str(doc.get("doc_id", doc.get("id", "")))
            if active_exclusion and did in active_exclusion:
                continue
            doc.update({"retrieval_score": float(score), "retrieval_rank": rank, "retrieval_type": "dense"})
            dense_docs.append(doc)
            rank += 1
            if len(dense_docs) >= fetch_k:
                break
        rrf_scores = defaultdict(float)
        doc_map = {}
        for r_list in (dense_docs, _bm25_results):
            for d in r_list:
                did = str(d.get("doc_id", d.get("id", "")))
                if not did or (active_exclusion and did in active_exclusion):
                    continue
                doc_map.setdefault(did, d)
                rrf_scores[did] += 1.0 / (60.0 + d.get("retrieval_rank", 100))
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        primary_docs = []
        seen_primary_titles = set()
        for did, score in sorted_rrf:
            doc = dict(doc_map[did])
            title_key = str(doc.get("title", "")).strip().casefold()
            if title_key and title_key in seen_primary_titles:
                continue
            if title_key:
                seen_primary_titles.add(title_key)
            rank = len(primary_docs) + 1
            doc.update({"retrieval_score": float(score), "retrieval_rank": rank, "retrieval_type": "hybrid"})
            primary_docs.append(doc)
            if len(primary_docs) >= primary_k:
                break

    search_timings["retrieval_primary.exclusion_filter"] = round(time.time() - t_excl0, 4)
    search_timings["retrieval_primary"] = round(time.time() - t0, 4)

    pass_results.append((f"primary [{primary_strategy}]", primary_docs))
    log.append({"pass": 1, "type": "primary", "strategy": primary_strategy,
                "query": variants[0], "documents_returned": len(primary_docs)})

    # Log sub-timings per question
    print(f"  [Primary Retrieval] Total: {search_timings['retrieval_primary']:.4f}s | Encode: {search_timings['retrieval_primary.query_encode']:.4f}s | FAISS: {search_timings['retrieval_primary.faiss_search']:.4f}s | BM25: {search_timings['retrieval_primary.bm25_search']:.4f}s | Filter: {search_timings['retrieval_primary.exclusion_filter']:.4f}s", flush=True)

    t_exp0 = time.time()
    for expanded in expand_query(query):
        if len(variants) >= MAX_QUERY_VARIANTS:
            break
        if expanded.casefold() not in {v.casefold() for v in variants}:
            variants.append(expanded)
    search_timings["expansion"] = round(time.time() - t_exp0, 4)

    t_dec0 = time.time()
    subqueries = decompose_query(query)
    for sq in subqueries:
        if len(variants) >= MAX_QUERY_VARIANTS:
            break
        if sq.casefold() not in {v.casefold() for v in variants}:
            variants.append(sq)
    search_timings["decomposition"] = round(time.time() - t_dec0, 4)

    t_var0 = time.time()
    for i, variant in enumerate(variants[1:], start=1):
        strategy = "HYBRID"
        run_pass("query_variant", variant, strategy, max(EXPANDED_TOP_K, top_k))
    search_timings["retrieval_variants"] = round(time.time() - t_var0, 4)

    t_cross0 = time.time()
    for strategy in ("DENSE", "HYBRID"):
        if strategy != primary_strategy:
            run_pass("cross_strategy", query, strategy, max(EXPANDED_TOP_K, top_k))
    search_timings["retrieval_cross_strategy"] = round(time.time() - t_cross0, 4)

    t_fuse0 = time.time()
    fused_docs = _merge_candidates(pass_results)
    final_docs = fused_docs[:max(top_k, MAX_CONTEXT_DOCS)]
    search_timings["retrieval_fusion"] = round(time.time() - t_fuse0, 4)

    return {
        "top_documents": final_docs,
        "all_candidates_count": len(fused_docs),
        "iterations_log": log,
        "is_exhausted": False,
        "full_corpus_scan_performed": False,
        "search_scope": "multi-query dense/sparse/hybrid candidate retrieval over the 250k corpus; not a literal scan of all 250k rows",
        "initial_strategy": primary_strategy,
        "query_variants": variants,
        "search_timings": search_timings
    }


@dataclass
class EvidenceAssessment:
    status: str
    directly_relevant: bool
    can_support_related_answer: bool
    missing_information: str
    reason: str = ""
    raw_response: str = ""
    json_parsed: bool = False
    parse_ok: bool = False
    fallback_fired: bool = False


def assess_evidence_sufficiency(query: str, documents: List[Dict[str, Any]]) -> EvidenceAssessment:
    if not documents:
        return EvidenceAssessment(
            status="INSUFFICIENT", directly_relevant=False, can_support_related_answer=False,
            missing_information="No documents retrieved.", reason="Empty document set.",
            raw_response="", json_parsed=True, parse_ok=True, fallback_fired=False
        )

    initial_docs = len(documents[:MAX_CONTEXT_DOCS])
    passages = []
    for i, doc in enumerate(documents[:MAX_CONTEXT_DOCS], 1):
        passages.append(
            f"[{i}] ID: {doc.get('doc_id', doc.get('id', 'unknown'))}\n"
            f"Title: {doc.get('title', 'Untitled')}\nContent: {_doc_text(doc)[:1000]}"
        )

    def _format_assess_prompt(passage_list: List[str]) -> str:
        return f"""Assess whether supplied evidence supports answering the question. Outside knowledge forbidden.
The answer may be inferred by combining facts from two or more documents. If the facts together answer the question, choose SUFFICIENT.
- SUFFICIENT: Evidence explicitly contains the answer or connecting facts.
- PARTIAL: Some facts supported, or related background is present.
- INSUFFICIENT: Documents are completely irrelevant and off-topic.

Question: {query}
Documents:
{chr(10).join(passage_list)}

Return JSON only:
{{"status":"SUFFICIENT|PARTIAL|INSUFFICIENT","direct":true|false,"related":true|false,"missing":"<max 12 words, or none>"}}"""

    prompt = _format_assess_prompt(passages)
    est_tokens = estimate_tokens(prompt)
    cut_count = 0
    while est_tokens > 1800 and len(passages) > 1:
        passages.pop()
        cut_count += 1
        prompt = _format_assess_prompt(passages)
        est_tokens = estimate_tokens(prompt)

    print(f"  [PromptGuard] Assessment: cut {cut_count} documents ({initial_docs} -> {len(passages)} kept). Prompt tokens: ~{est_tokens}", flush=True)

    # First attempt with num_predict=200
    res = call_gemma(prompt, json_mode=True, timeout=300, num_predict=200)

    # If JSON parsing failed, retry once with a higher limit (num_predict=320)
    if not (res["success"] and isinstance(res.get("json"), dict)):
        print(f"  [CARO Assessment] First JSON parse failed/truncated. Retrying once with num_predict=320...", flush=True)
        res = call_gemma(prompt, json_mode=True, timeout=300, num_predict=320)

    raw_resp = str(res.get("raw", "")).strip()

    if res["success"] and isinstance(res["json"], dict):
        parsed = res["json"]
        status = str(parsed.get("status", "INSUFFICIENT")).upper().strip()
        if status not in {"SUFFICIENT", "PARTIAL", "INSUFFICIENT"}:
            status = "INSUFFICIENT"
        direct = bool(parsed.get("direct", parsed.get("directly_relevant", False)))
        related = bool(parsed.get("related", parsed.get("can_support_related_answer", False)))
        # Evidence across multiple documents collectively satisfies or partially satisfies the question.
        # Preserve status directly as evaluated by the model.
        missing = str(parsed.get("missing", parsed.get("missing_information", "None"))).strip()
        missing_words = missing.split()
        if len(missing_words) > 12:
            missing = " ".join(missing_words[:12])

        print(f"  [CARO Assessment] raw='{raw_resp[:120]}' | parse_ok=True | fallback_fired=False | status={status}", flush=True)
        return EvidenceAssessment(
            status=status, directly_relevant=direct,
            can_support_related_answer=related,
            missing_information=missing,
            reason=str(parsed.get("reason", "")),
            raw_response=raw_resp,
            json_parsed=True,
            parse_ok=True,
            fallback_fired=False
        )

    print(f"  [CARO Assessment] raw='{raw_resp[:120]}' | parse_ok=False | fallback_fired=True (Conservative Fallback) | error='{res.get('error')}'", flush=True)
    return EvidenceAssessment(
        status="INSUFFICIENT",
        directly_relevant=False,
        can_support_related_answer=False,
        missing_information="Evidence assessment failed; support could not be verified.",
        reason=f"Conservative fallback because evaluator failed: {res.get('error')}",
        raw_response=raw_resp,
        json_parsed=False,
        parse_ok=False,
        fallback_fired=True
    )


class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []

    def add_turn(self, user_text: str, assistant_text: str):
        self.turns.append({"role": "user", "content": str(user_text).strip()})
        self.turns.append({"role": "assistant", "content": str(assistant_text).strip()})
        keep_messages = self.max_turns * 2
        if len(self.turns) > keep_messages:
            self.turns = self.turns[-keep_messages:]

    def get_turns(self) -> List[Dict[str, str]]:
        return list(self.turns)

    def clear(self):
        self.turns = []


def generate_grounded_answer(
    query: str,
    documents: List[Dict[str, Any]],
    is_partial: bool = False,
    missing_info: str = ""
) -> str:
    if not documents:
        return UNANSWERABLE_STRING
    from eval.common import build_shared_generation_prompt
    prompt = build_shared_generation_prompt(
        query=query,
        documents=documents,
        system_type="caro",
        is_partial=False,
        missing_info=""
    )
    res = call_gemma(prompt, json_mode=False, timeout=300, max_retries=2, num_predict=64)
    if not res["success"] or not res["raw"].strip():
        return UNANSWERABLE_STRING
    return res["raw"].strip()


class CaroRAGPipeline:
    def __init__(self):
        self.memory = ConversationMemory(max_turns=5)

    def clear_memory(self):
        self.memory.clear()

    def query(self, user_query: str) -> Dict[str, Any]:
        start_time = time.time()
        user_query = str(user_query).strip()
        stage_timings: Dict[str, float] = {
            "router": 0.0,
            "retrieval_primary": 0.0,
            "retrieval_primary.query_encode": 0.0,
            "retrieval_primary.faiss_search": 0.0,
            "retrieval_primary.bm25_search": 0.0,
            "retrieval_primary.exclusion_filter": 0.0,
            "expansion": 0.0,
            "decomposition": 0.0,
            "retrieval_variants": 0.0,
            "retrieval_cross_strategy": 0.0,
            "retrieval_fusion": 0.0,
            "assessment": 0.0,
            "generation": 0.0,
            "total": 0.0
        }
        if not user_query:
            stage_timings["total"] = round(time.time() - start_time, 4)
            return {
                "query": user_query, "route": "DIRECT", "strategy": "DIRECT", "iterations_run": 0,
                "retrieved_documents": [], "evidence_status": "NOT_APPLICABLE",
                "final_answer": "Please enter a question.", "disclaimer": None,
                "latency_seconds": round(time.time() - start_time, 3),
                "stage_timings": stage_timings
            }

        history = self.memory.get_turns()
        t_router0 = time.time()
        routing = route_query_with_llm(user_query, history)
        stage_timings["router"] = round(time.time() - t_router0, 4)

        if routing.route == "DIRECT":
            t_gen0 = time.time()
            gen_res = call_gemma(
                f"Answer the following conversational request directly. Do not claim to have searched a corpus.\n{user_query}",
                timeout=300,
                num_predict=64
            )
            final_answer = gen_res["raw"].strip() if gen_res["success"] and gen_res["raw"].strip() else "I could not generate a response right now. Please retry."
            stage_timings["generation"] = round(time.time() - t_gen0, 4)
            stage_timings["total"] = round(time.time() - start_time, 4)
            self.memory.add_turn(user_query, final_answer)
            return {
                "query": user_query, "route": "DIRECT", "strategy": "DIRECT", "iterations_run": 0,
                "retrieved_documents": [], "evidence_status": "NOT_APPLICABLE",
                "final_answer": final_answer, "disclaimer": None,
                "latency_seconds": round(time.time() - start_time, 3),
                "stage_timings": stage_timings
            }

        if routing.route == "MEMORY" and not routing.requires_retrieval:
            mem_prompt = f"""Answer only from this conversation history.
If the requested fact is not present in the history, return exactly NEEDS_RETRIEVAL.
Do not guess.
History: {json.dumps(history, ensure_ascii=False)}
Question: {user_query}
Answer:"""
            t_mem0 = time.time()
            mem_res = call_gemma(mem_prompt, timeout=300, num_predict=64)
            memory_answer = mem_res["raw"].strip() if mem_res["success"] else "NEEDS_RETRIEVAL"
            stage_timings["generation"] = round(time.time() - t_mem0, 4)
            if memory_answer and "NEEDS_RETRIEVAL" not in memory_answer.upper():
                stage_timings["total"] = round(time.time() - start_time, 4)
                self.memory.add_turn(user_query, memory_answer)
                return {
                    "query": user_query, "route": "MEMORY", "strategy": "MEMORY", "iterations_run": 0,
                    "retrieved_documents": [], "evidence_status": "RESOLVED_FROM_MEMORY",
                    "final_answer": memory_answer, "disclaimer": None,
                    "latency_seconds": round(time.time() - start_time, 3),
                    "stage_timings": stage_timings
                }

        strategy_info = select_retrieval_strategy(routing)
        active_strategy = strategy_info["strategy"]
        if active_strategy in ("DIRECT", "MEMORY"):
            active_strategy = "DENSE"
        elif active_strategy == "MEMORY_AND_RETRIEVAL":
            active_strategy = "HYBRID"

        effective_query = routing.query_rewritten.strip() if routing.query_rewritten.strip() else user_query
        search_result = execute_full_corpus_search(
            query=effective_query,
            initial_strategy=active_strategy,
            min_confidence_score=0.55,
            top_k=MAX_CONTEXT_DOCS
        )
        search_timings = search_result.get("search_timings", {})
        stage_timings.update(search_timings)

        retrieved_docs = search_result["top_documents"]

        t_assess0 = time.time()
        assessment = assess_evidence_sufficiency(effective_query, retrieved_docs)
        stage_timings["assessment"] = round(time.time() - t_assess0, 4)

        disclaimer = None
        t_gen0 = time.time()
        if assessment.status == "SUFFICIENT":
            final_answer = generate_grounded_answer(effective_query, retrieved_docs, is_partial=False)
            stage_timings["generation"] = round(time.time() - t_gen0, 4)
        elif assessment.status == "PARTIAL":
            disclaimer = f"Partial evidence; missing: {assessment.missing_information}"
            final_answer = generate_grounded_answer(
                effective_query, retrieved_docs, is_partial=True,
                missing_info=assessment.missing_information
            )
            stage_timings["generation"] = round(time.time() - t_gen0, 4)
        else:
            if retrieved_docs:
                disclaimer = f"Partial evidence; missing: {assessment.missing_information}"
                final_answer = generate_grounded_answer(
                    effective_query, retrieved_docs, is_partial=True,
                    missing_info=assessment.missing_information or "Evidence details partially missing"
                )
                stage_timings["generation"] = round(time.time() - t_gen0, 4)
            else:
                final_answer = UNANSWERABLE_STRING
                stage_timings["generation"] = 0.0

        elapsed = time.time() - start_time
        stage_timings["total"] = round(elapsed, 4)
        self.memory.add_turn(user_query, final_answer)
        return {
            "query": user_query,
            "rewritten_query": effective_query,
            "route": "RETRIEVE" if routing.route == "MEMORY" else routing.route,
            "strategy": active_strategy,
            "iterations_run": len(search_result["iterations_log"]),
            "search_scope": search_result["search_scope"],
            "full_corpus_scan_performed": search_result["full_corpus_scan_performed"],
            "retrieval_passes": search_result["iterations_log"],
            "candidate_count": search_result["all_candidates_count"],
            "retrieved_documents": retrieved_docs,
            "evidence_status": assessment.status,
            "evidence_assessment": asdict(assessment),
            "final_answer": final_answer,
            "disclaimer": disclaimer,
            "latency_seconds": round(elapsed, 3),
            "stage_timings": stage_timings
        }
