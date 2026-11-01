import os
import threading
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config.config import EMBEDDING_MODEL_NAME


def configure_cpu_threads(num_threads: int | None = None) -> int:
    """Configures PyTorch CPU execution threads for optimal i5-10300H performance."""
    if num_threads is not None and num_threads > 0:
        torch.set_num_threads(num_threads)
    return torch.get_num_threads()


def get_default_device() -> str:
    """Returns 'cuda' if a compatible GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingModel:
    """Reusable embedding component for document and query encoding."""

    _model_cache: dict[str, SentenceTransformer] = {}
    _cache_lock: threading.Lock = threading.Lock()

    def __init__(self, model_name: str | None = None, device: str | None = None, num_threads: int | None = None):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        if device is None or device == "auto":
            self.device = get_default_device()
        else:
            self.device = device

        if self.device == "cpu" and num_threads is not None:
            configure_cpu_threads(num_threads)
        cache_key = f"{self.model_name}:{self.device}"
        with EmbeddingModel._cache_lock:
            if cache_key not in EmbeddingModel._model_cache:
                EmbeddingModel._model_cache[cache_key] = SentenceTransformer(self.model_name, device=device)
            self.model = EmbeddingModel._model_cache[cache_key]

    def encode_documents(self, documents, batch_size: int = 32, show_progress_bar: bool = False):
        if not documents:
            return np.empty((0, 0), dtype=np.float32)
        embeddings = self.model.encode(
            list(documents),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str):
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding[0], dtype=np.float32)
