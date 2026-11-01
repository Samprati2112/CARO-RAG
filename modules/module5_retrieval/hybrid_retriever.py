from __future__ import annotations

from typing import Any

from .schemas import RetrievalConfig


class HybridRetriever:
    """Fuse dense and BM25 rankings with deterministic min-max score fusion."""

    def __init__(self, dense_retriever, sparse_retriever, config: RetrievalConfig | None = None, dense_score_model=None, sparse_score_model=None):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.config = config or RetrievalConfig()
        self.dense_score_model = dense_score_model
        self.sparse_score_model = sparse_score_model

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        limit = top_k or self.config.retrieval_top_k
        # --- BUGGED ORIGINAL (Uncomment to revert) ---
        # dense_results = self.dense_retriever.retrieve(query, limit)
        # sparse_results = self.sparse_retriever.retrieve(query, limit)
        # dense_scores = {self._document_id(document): float(document["retrieval_score"]) for document in dense_results}
        # sparse_scores = {self._document_id(document): float(document["retrieval_score"]) for document in sparse_results}
        # all_ids = set(dense_scores) | set(sparse_scores)
        # by_id = {self._document_id(document): document for document in dense_results + sparse_results}
        # fused = []
        # for document_id in all_ids:
        #     document = dict(by_id[document_id])
        #     dense_score = dense_scores.get(document_id, 0.0)
        #     sparse_score = sparse_scores.get(document_id, 0.0)
        #     normalized_dense_score = self.dense_score_model.transform(dense_score) if document_id in dense_scores and self.dense_score_model else dense_score
        #     normalized_sparse_score = self.sparse_score_model.transform(sparse_score) if document_id in sparse_scores and self.sparse_score_model else sparse_score
        #     fusion_score = (
        #         self.config.dense_weight * normalized_dense_score
        #         + self.config.sparse_weight * normalized_sparse_score
        #     )
        #     document.update({
        #         "document_id": document_id,
        #         "dense_score": dense_score,
        #         "sparse_score": sparse_score,
        #         "normalized_dense_score": normalized_dense_score,
        #         "normalized_sparse_score": normalized_sparse_score,
        #         "fusion_score": fusion_score,
        #         "retrieval_score": fusion_score,
        #     })
        #     fused.append(document)
        # fused.sort(key=lambda document: (-document["fusion_score"], document["document_id"]))
        # for rank, document in enumerate(fused[:limit], start=1):
        #     document["retrieval_rank"] = rank
        # return fused[:limit]
        # --- END BUGGED ORIGINAL ---

        # --- FIX: Broader candidate pool before fusion and avoid halving single-modality scores ---
        candidate_pool_limit = max(limit * 4, 20)
        dense_results = self.dense_retriever.retrieve(query, candidate_pool_limit)
        sparse_results = self.sparse_retriever.retrieve(query, candidate_pool_limit)
        dense_scores = {self._document_id(document): float(document["retrieval_score"]) for document in dense_results}
        sparse_scores = {self._document_id(document): float(document["retrieval_score"]) for document in sparse_results}
        all_ids = set(dense_scores) | set(sparse_scores)
        by_id = {self._document_id(document): document for document in dense_results + sparse_results}
        fused = []
        for document_id in all_ids:
            document = dict(by_id[document_id])
            dense_score = dense_scores.get(document_id, 0.0)
            sparse_score = sparse_scores.get(document_id, 0.0)
            normalized_dense_score = self.dense_score_model.transform(dense_score) if document_id in dense_scores and self.dense_score_model else dense_score
            normalized_sparse_score = self.sparse_score_model.transform(sparse_score) if document_id in sparse_scores and self.sparse_score_model else sparse_score

            # If present in both, blend weights. If present in only one, avoid penalizing to zero for the missing modality.
            if document_id in dense_scores and document_id in sparse_scores:
                fusion_score = (
                    self.config.dense_weight * normalized_dense_score
                    + self.config.sparse_weight * normalized_sparse_score
                )
            elif document_id in dense_scores:
                fusion_score = normalized_dense_score
            else:
                fusion_score = normalized_sparse_score

            document.update({
                "document_id": document_id,
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "normalized_dense_score": normalized_dense_score,
                "normalized_sparse_score": normalized_sparse_score,
                "fusion_score": fusion_score,
                "retrieval_score": fusion_score,
            })
            fused.append(document)
        fused.sort(key=lambda document: (-document["fusion_score"], document["document_id"]))
        for rank, document in enumerate(fused[:limit], start=1):
            document["retrieval_rank"] = rank
        return fused[:limit]
        # --- END FIX ---

    @staticmethod
    def _document_id(document: dict[str, Any]) -> str:
        return str(document.get("document_id") or document.get("doc_id") or document.get("id"))
