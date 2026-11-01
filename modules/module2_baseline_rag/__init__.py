"""Baseline RAG module for dense retrieval and generation."""

from .embeddings import EmbeddingModel
from .generator import Generator
from .pipeline import BaselineRAGPipeline
from .retriever import DenseRetriever
from .vector_store import FAISSVectorStore

__all__ = [
    "EmbeddingModel",
    "Generator",
    "BaselineRAGPipeline",
    "DenseRetriever",
    "FAISSVectorStore",
]
