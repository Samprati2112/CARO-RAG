from __future__ import annotations

from config.config import DEFAULT_TOP_K, HOTPOTQA_DOCS_OUTPUT, HOTPOTQA_INDEX_PATH, MAX_INDEX_DOCUMENTS
from modules.module1_data_processing.processor import read_jsonl
from modules.module2_baseline_rag.embeddings import EmbeddingModel
from modules.module2_baseline_rag.retriever import DenseRetriever
from modules.module2_baseline_rag.vector_store import FAISSVectorStore


class BaselineRAGPipeline:
    """Standard dense-retrieval baseline without adaptive logic."""

    def __init__(
        self,
        vector_store: FAISSVectorStore | None = None,
        retriever: DenseRetriever | None = None,
        generator=None,
        embedding_model: EmbeddingModel | None = None,
        top_k: int = DEFAULT_TOP_K,
        index_path: str | None = None,
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or FAISSVectorStore(index_path or HOTPOTQA_INDEX_PATH, top_k=top_k)
        self.retriever = retriever or DenseRetriever(self.vector_store, top_k=top_k)
        self.generator = generator
        self.top_k = top_k

    def load_documents(self, path: str = str(HOTPOTQA_DOCS_OUTPUT), limit: int | None = None):
        records = read_jsonl(path)
        if limit is not None:
            return records[:limit]
        return records

    def build_index(self, documents=None, path: str | None = None, limit: int | None = MAX_INDEX_DOCUMENTS):
        source_docs = documents if documents is not None else self.load_documents(limit=limit)
        if not source_docs:
            raise ValueError("No processed HotPotQA documents were found to index.")
        print(f"[1/5] Loading documents... using {len(source_docs)} of the processed corpus.")
        self.vector_store.build_from_documents(source_docs, self.embedding_model, limit=len(source_docs))
        if path is not None:
            self.vector_store.index_path = path
            self.vector_store.metadata_path = self.vector_store.index_path.with_suffix(".meta.json")
        print("[3/5] Building FAISS index...")
        self.vector_store.save()
        print(f"[5/5] FAISS index saved to {self.vector_store.index_path}")
        return self.vector_store

    def query(self, question: str, top_k: int | None = None):
        if self.vector_store.index is None:
            if not self.vector_store.index_path.exists():
                print("Index not found. Building a development FAISS index first...")
                self.build_index(limit=MAX_INDEX_DOCUMENTS)
            print("[4/5] Loading existing FAISS index...")
            self.vector_store.load()
        if self.vector_store.index is None:
            raise ValueError("FAISS index is unavailable. Build the index first.")
        print(f"[4/5] Retrieving Top-{top_k or self.top_k} documents...")
        query_vector = self.embedding_model.encode_query(question)
        return self.retriever.search(query_vector, top_k=top_k or self.top_k)

    def answer_query(self, question: str, top_k: int | None = None):
        results = self.query(question, top_k=top_k)
        context = [doc for doc in results] if results else []
        print("Context constructed successfully.")
        if self.generator is None:
            raise ValueError("A generator instance is required to answer the query.")
        print("[5/5] Generating answer with Ollama / Gemma 3 4B...")
        return self.generator.generate(question, context)

    def build_and_answer(self, question: str, documents=None):
        if self.vector_store.index is None:
            self.build_index(documents)
        return self.answer_query(question)
