# Module 5: Retrieval and Relevance Filtering

Module 5 is the CARO evidence execution layer. It consumes the structured
`RetrievalStrategyResult` selected by Module 4, retrieves candidate documents,
and applies deterministic score-based filtering. It does not generate answers,
store memory, perform iterative retrieval, or call Gemma/Ollama.

## Responsibility boundaries

- M3 asks: **What does the query need?**
- M4 asks: **How should it be retrieved?**
- M5 asks: **Which retrieved evidence is relevant enough to pass downstream?**

Module 5 does not repeat Module 3 routing or Module 4 strategy rules.
`MEMORY` returns without external retrieval so Module 6 can resolve context.
`MEMORY_AND_RETRIEVAL` executes the external retrieval portion only; memory is
left to Module 6.

## Retrieval modes

- `DENSE`: reuses Module 2's BGE-M3 `EmbeddingModel` and `FAISSVectorStore`.
  The configured development index is `indexes/hotpotqa/hotpotqa_dev_index.faiss`.
  Query execution never builds an index.
- `SPARSE`: uses a deterministic in-project BM25 implementation over the
  standardized HotPotQA document JSONL. The default corpus is capped at the
  5,000-document development subset, and statistics are cached in
  `data/processed/hotpotqa_dev_bm25.pkl` when possible.
- `HYBRID`: retrieves from both dense and BM25, transforms each raw modality
  score using its development-fitted global score model, and combines scores as:

  `fusion_score = dense_weight * normalized_dense_score + sparse_weight * normalized_sparse_score`

  The default weights are `0.5` and `0.5`. Documents are merged by document ID
  and ranked by descending fusion score, with document ID as a deterministic tie-breaker.

Dense and sparse raw scores are not assumed to be comparable. M5 therefore does
not use per-query min-max normalization: that procedure forces the current
query's maximum to 1.0 and makes a global threshold query-dependent. Instead,
development calibration fits a deterministic regularized logistic transform for
each modality:

`p(s) = 1 / (1 + exp(-(a * ((s - mu) / sigma) + b)))`

where `s` is the raw score and `mu`, `sigma`, `a`, and `b` are fitted once from
pooled labeled development candidates. Dense and sparse calibrated scores are
then fused, preserving raw scores and calibrated component scores in output.

`DIRECT` returns a structured zero-candidate result. `MEMORY` also returns zero
candidates and explicitly reports that external retrieval was not executed.

## Score-based filtering

Filtering uses `normalized_score` (the fusion score for hybrid retrieval, or the
method-specific normalized score for dense and sparse retrieval), a
configurable `relevance_threshold`, and configurable `final_top_n`.

When `relevance_threshold` is `None`, M5 calibrates the threshold
reproducibly from the processed HotPotQA development corpus. It uses only
questions whose supporting documents occur within the first 5,000 processed
documents, retrieves candidates for a deterministic question-ID-sorted subset,
and labels a candidate positive only when its document ID matches a
HotPotQA `metadata.is_supporting` document. Candidate thresholds are the
observed scores. M5 selects the threshold maximizing micro-F1 over retrieved
candidates; ties are resolved by choosing the highest threshold, then higher
precision and recall. The selected threshold and calibration metrics are saved
to `indexes/hotpotqa/m5_calibration.json` and included in the result.

Calibration is performed separately for `DENSE`, `SPARSE`, and `HYBRID`.
`MEMORY_AND_RETRIEVAL` uses the dense external-retrieval calibration while
leaving memory to Module 6. This is a score-based filtering criterion, not a
claim of semantic truth, human relevance classification, or learned
cross-encoder reranking. The calibration data are development data only and
must not be treated as final evaluation results.

The current artifact is calculated from 128 queries and 1,280 retrieved
candidates per strategy over the 5,000-document development corpus:

| Strategy | Threshold | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| `DENSE` | 0.2371527091 | 0.5589 | 0.6917 | 0.6182 |
| `SPARSE` | 0.2298907963 | 0.3826 | 0.4055 | 0.3937 |
| `HYBRID` | 0.1871744086 | 0.5124 | 0.6735 | 0.5820 |

These are calibration-set measurements, not final test-set performance claims.

The filter keeps candidates with scores at or above the calibrated threshold,
then applies `final_top_n`. Every returned candidate records whether it was
retained and whether it was removed by the relevance criterion.

## Deduplication

Before filtering, M5 removes exact duplicate candidate content after whitespace
normalization and case folding. The first ranked occurrence is retained;
duplicate document IDs are reported in `duplicate_document_ids` and counted in
`deduplicated_count`. Titles alone are never used as a duplicate key, so
different documents sharing a title are not merged.

The result distinguishes `candidate_count` before deduplication,
`unique_candidate_count` after deduplication, `relevant_count` retained after
score filtering, and each document's `filter_reason`.

## Development evaluation

`evaluation.py` provides development-only metrics: supporting-document
recall, retained-supporting precision, candidate and retained counts,
percentage filtered, and duplicate count. A supporting document means a
document marked by the HotPotQA annotation, not a semantic judgment made by
M5.

Results retain candidate count, unique candidate count, selected evidence
count, document IDs, text, ranks, retrieval scores, normalized scores, and
dense/sparse/fusion metadata where applicable. Downstream modules may consume
the selected evidence, but M5 itself does not implement memory, iterative
retrieval, evidence sufficiency checking, or generation.

## CLI

```text
python main.py --module 5 --action retrieve --dataset hotpotqa --question "What is the population of Reykjavik?"
```

Dense and hybrid commands require the existing Module 2 runtime dependencies
and development FAISS index. Module 5 does not silently build a missing index
or embed the full HotPotQA corpus.

## Limitations

The implementation uses retrieval scores as a deterministic filtering signal.
It does not claim those scores are human annotations. It does not include
cross-encoder reranking, learned relevance classification, answer synthesis,
memory access, evidence sufficiency checks, or iterative retrieval. Those
concerns belong to later modules or future evaluated extensions.
