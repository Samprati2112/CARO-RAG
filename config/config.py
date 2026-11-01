from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MEMORY_DIR = DATA_DIR / "memory"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEXES_DIR = PROJECT_ROOT / "indexes"
HOTPOTQA_INDEX_DIR = INDEXES_DIR / "hotpotqa"
QRECC_INDEX_DIR = INDEXES_DIR / "qrecc"

HOTPOTQA_RAW_DIR = RAW_DATA_DIR / "hotpotqa"
QRECC_RAW_DIR = RAW_DATA_DIR / "qrecc"

HOTPOTQA_DOCS_OUTPUT = PROCESSED_DATA_DIR / "hotpotqa_documents.jsonl"
HOTPOTQA_QUESTIONS_OUTPUT = PROCESSED_DATA_DIR / "hotpotqa_questions.jsonl"
QRECC_DOCS_OUTPUT = PROCESSED_DATA_DIR / "qrecc_documents.jsonl"
QRECC_QUESTIONS_OUTPUT = PROCESSED_DATA_DIR / "qrecc_questions.jsonl"
QRECC_CONVERSATIONS_OUTPUT = PROCESSED_DATA_DIR / "qrecc_conversations.jsonl"

# Development indexes and caches (preserved)
HOTPOTQA_DEV_INDEX_PATH = HOTPOTQA_INDEX_DIR / "hotpotqa_dev_index.faiss"
HOTPOTQA_DEV_BM25_PATH = PROCESSED_DATA_DIR / "hotpotqa_dev_bm25.pkl"
HOTPOTQA_INDEX_PATH = HOTPOTQA_DEV_INDEX_PATH

# Full-corpus indexes and caches (HotPotQA)
HOTPOTQA_FULL_INDEX_PATH = HOTPOTQA_INDEX_DIR / "hotpotqa_full_index.faiss"
HOTPOTQA_FULL_INDEX_META_PATH = HOTPOTQA_INDEX_DIR / "hotpotqa_full_index.meta.json"
HOTPOTQA_FULL_BM25_PATH = HOTPOTQA_INDEX_DIR / "hotpotqa_full_bm25.pkl"
HOTPOTQA_BM25_CACHE_PATH = HOTPOTQA_INDEX_DIR / "bm25_cache.npz"

# Full-corpus indexes and caches (QReCC)
QRECC_DEV_INDEX_PATH = QRECC_INDEX_DIR / "qrecc_dev_index.faiss"
QRECC_DEV_BM25_PATH = QRECC_INDEX_DIR / "qrecc_dev_bm25.pkl"
QRECC_FULL_INDEX_PATH = QRECC_INDEX_DIR / "qrecc_full_index.faiss"
QRECC_FULL_INDEX_META_PATH = QRECC_INDEX_DIR / "qrecc_full_index.meta.json"
QRECC_FULL_BM25_PATH = QRECC_INDEX_DIR / "qrecc_full_bm25.pkl"

HOTPOTQA_DATASET_NAMES = ["hotpotqa/hotpot_qa"]
HOTPOTQA_CONFIG_CANDIDATES = ["distractor", "fullwiki"]
QRECC_DATASET_NAMES = ["svakulenk0/qrecc"]

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL_NAME = "gemma3:4b"
DEFAULT_TOP_K = 5
MAX_SAMPLE_RECORDS = 3
MAX_CONTEXT_LENGTH = 2000
MAX_INDEX_DOCUMENTS = 5000
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "gemma3:4b"

# Indexing batch controls
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_INDEXING_CHUNK_SIZE = 1000

