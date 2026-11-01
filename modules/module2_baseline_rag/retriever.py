from __future__ import annotations

import numpy as np


class DenseRetriever:
    """Simple dense retrieval over a FAISS index."""

    def __init__(self, vector_store, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    def search(self, query_vector: np.ndarray, top_k: int | None = None):
        requested_k = top_k if top_k is not None else self.top_k
        return self.vector_store.search(query_vector, top_k=requested_k)
