# Module 4: Query Classification & Retrieval Strategy Selection

Module 4 answers how information should be retrieved after Module 3 has
answered what the query requires. It consumes Module 3's structured
`RoutingResult` and produces a deterministic `RetrievalStrategyResult`.

## Supported strategies

- `DIRECT`: no external retrieval is required.
- `DENSE`: semantic retrieval using the BGE-M3 and FAISS infrastructure from Module 2.
- `SPARSE`: lexical retrieval using BM25. Module 4 only selects it.
- `HYBRID`: dense plus sparse retrieval, useful for multi-hop queries. Module 4 only selects it.
- `MEMORY`: conversational context is required without external retrieval.
- `MEMORY_AND_RETRIEVAL`: both conversational context and external knowledge are required.

## Selection rules

1. Neither retrieval nor memory required -> `DIRECT`.
2. Retrieval and memory both required -> `MEMORY_AND_RETRIEVAL`.
3. Memory only -> `MEMORY`.
4. Multi-hop retrieval -> `HYBRID`.
5. Named-entity queries with exact lookup terms -> `SPARSE`.
6. Other retrieval-required queries -> `DENSE`.

Complexity and memory dependency remain separate. A multi-hop query does not
imply memory, and a comparison query does not imply memory.

## Decision process

Module 4 asks: **How should retrieval be performed?** It consumes the complete
Module 3 `RoutingResult`, including its question type, retrieval requirement,
memory requirement, and query characteristics. It does not redo Module 3's
route decision or collapse routing and strategy selection into one module.

The priority order is:

1. No retrieval and no memory -> `DIRECT`.
2. Retrieval and memory -> `MEMORY_AND_RETRIEVAL`.
3. Memory only -> `MEMORY`.
4. Multi-hop retrieval -> `HYBRID`.
5. Retrieval with exact-keyword-sensitive terms and a named entity -> `SPARSE`.
6. Other retrieval-required queries -> `DENSE`.

Each result reason states the M3 question type, relevant signals, and why the
selected technique fits. Procedural and comparison queries use `DENSE` when
they do not have the multi-hop or exact lexical signals that would select a
more specialized strategy. A multi-hop query uses `HYBRID` because evidence
may be associated with multiple entities or relationships.

Module 4 confidence values are also heuristic rule-strength scores, not
calibrated probabilities.

The rules are deterministic, lightweight, explainable, and independent of LLM
generation variability. Module 4 does not perform FAISS, BM25, hybrid fusion,
memory access, reranking, or answer generation. Module 5 will consume
`selected_strategy` and perform retrieval and relevance processing.

## CLI

```text
python main.py --module 4 --action strategy --question "What is Python?"
```