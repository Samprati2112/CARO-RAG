# Module 6: Temporary Memory Management

Module 6 stores and retrieves temporary task/conversation context. It consumes a
query and conversation identifier and returns prior turns when semantic
similarity passes a configurable threshold.

## Boundaries

- M3 decides whether memory is required.
- M4 selects `MEMORY` or `MEMORY_AND_RETRIEVAL` when appropriate.
- M5 retrieves external evidence.
- M6 retrieves temporary conversational context only.

M6 never performs external document retrieval, generation, iterative retrieval,
evidence sufficiency checking, or LLM-based relevance decisions. For
`MEMORY_AND_RETRIEVAL`, M6 retrieves memory only; M5 remains independent.

## Storage and embeddings

The default local store is `data/memory/memory_records.json`. Each record keeps
its conversation and turn identifiers, original query, response, context,
metadata, timestamp, and a normalized BGE-M3 embedding. The existing Module 2
`EmbeddingModel` is loaded lazily and the memory embeddings are stored in this
separate file, never in the document FAISS index.

The store is temporary and easy to reset with `clear_memory()`. Retrieval first
filters by `conversation_id`, so unrelated conversations cannot mix.

Similarity is cosine similarity because stored and query vectors are L2
normalized:

`similarity(q, m) = q dot m`

Results are sorted by descending similarity, then turn ID and memory ID for
deterministic ties. `memory_top_k` and `memory_relevance_threshold` are
configurable. A score is a retrieval signal, not a human relevance annotation.

## API

```python
from modules.module6_temporary_memory import TemporaryMemoryManager

memory = TemporaryMemoryManager(
    storage_path="data/memory",
    memory_top_k=5,
    memory_relevance_threshold=0.35,
)
memory.add_memory(
    conversation_id="demo",
    turn_id="1",
    user_query="What is REST API?",
    assistant_response="REST API is an architectural style for web services.",
    context="REST API architectural style and HTTP resources.",
)
result = memory.retrieve_memory("What are its advantages?", "demo")
```

Every retrieved item exposes `memory_id`, `conversation_id`, `turn_id`,
`similarity`, original query, stored response/context, timestamp, metadata, and
retrieval rank.

## CLI

```text
python main.py --module 6 --action add --conversation-id demo --turn-id 1 --question "What is REST API?" --response "REST API is an architectural style for web services."
python main.py --module 6 --action retrieve --conversation-id demo --question "What are its advantages?"
python main.py --module 6 --action clear --conversation-id demo
```

The CLI uses the local BGE-M3 environment and does not call Gemma/Ollama.
